#include "mtl_pipeline_cache.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <libgen.h>
#include <mach-o/dyld.h>
#include <sys/stat.h>

extern "C" void host_stall_hook(int begin) __attribute__((weak));

namespace mtl_d3d9 {

static const char *kHelperShadersMsl = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct BlitParams {
    float4 srcRect; /* u0, v0, uSize, vSize */
    float4 dstRect; /* ndcX0, ndcY0, ndcW, ndcH (top-left origin mapped to NDC) */
};

struct BlitOut {
    float4 pos [[position]];
    float2 uv;
};

vertex BlitOut mtl_blit_vs(uint vid [[vertex_id]], constant BlitParams &p [[buffer(0)]]) {
    const float2 corners[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0),
    };
    float2 c = corners[vid & 3u];
    BlitOut o;
    float x = p.dstRect.x + c.x * p.dstRect.z;
    float y = p.dstRect.y + c.y * p.dstRect.w;
    o.pos = float4(x, y, 0.0, 1.0);
    o.uv  = p.srcRect.xy + c * p.srcRect.zw;
    return o;
}

fragment float4 mtl_blit_ps(BlitOut in [[stage_in]],
                            texture2d<float> tex [[texture(0)]],
                            sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.uv);
}

fragment half4 mtl_blit_ps_half(BlitOut in [[stage_in]],
                                texture2d<half> tex [[texture(0)]],
                                sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.uv);
}

struct ClearParams {
    float4 dstRect; /* ndcX0, ndcY0, ndcW, ndcH */
    float4 color;
    float  depth;
    float  _pad[3];
};

vertex float4 mtl_clear_vs(uint vid [[vertex_id]], constant ClearParams &p [[buffer(0)]]) {
    const float2 corners[4] = {
        float2(0.0, 0.0),
        float2(1.0, 0.0),
        float2(0.0, 1.0),
        float2(1.0, 1.0),
    };
    float2 c = corners[vid & 3u];
    float x = p.dstRect.x + c.x * p.dstRect.z;
    float y = p.dstRect.y + c.y * p.dstRect.w;
    return float4(x, y, p.depth, 1.0);
}

fragment half4 mtl_clear_ps(constant ClearParams &p [[buffer(0)]]) {
    return half4(p.color);
}
)MSL";

struct MtlPsoManifestHeader {
    uint32_t magic = 0x4D505331u; /* 'MPS1' */
    uint32_t version = 1u;
    uint32_t recordSize = uint32_t(sizeof(MtlPsoManifestRecord));
    uint32_t count = 0;
};

static std::atomic<uint64_t> gPsoTmpSeq{0};

static std::string pipelineCacheModeSuffix() {
    const char *se = getenv("ME_PC_SAFE_MATH");
    const char *ee = getenv("ME_PC_EMULATE_DREF");
    bool safe = (se && *se == '1');
    bool emuDref = (ee && *ee == '1');
    if (safe && emuDref) return "se";
    if (safe) return "s";
    if (emuDref) return "e";
    return "";
}

static MtlShaderVariantKeyPod packPsVariantKey(const MtlShaderVariantKey &k) {
    MtlShaderVariantKeyPod p = {};
    p.vsIoHash            = k.vsIoHash;
    p.samplerDepthCmpMask = k.samplerDepthCmpMask;
    p.samplerNullMask     = k.samplerNullMask;
    p.samplerProjMask     = k.samplerProjMask;
    p.samplerGatherMask   = k.samplerGatherMask;
    p.colorWriteMask      = k.colorWriteMask;
    p.flags = (k.hasDepthAttach ? 1u : 0u) |
              (k.flatShade ? 2u : 0u) |
              (k.alphaTestEnable ? 4u : 0u) |
              (k.pointSpriteEnable ? 8u : 0u);
    return p;
}

static MtlShaderVariantKey unpackPsVariantKey(const MtlShaderVariantKeyPod &p) {
    MtlShaderVariantKey k = {};
    k.vsIoHash            = p.vsIoHash;
    k.colorWriteMask      = p.colorWriteMask;
    k.hasDepthAttach      = (p.flags & 1u) != 0u;
    k.flatShade           = (p.flags & 2u) != 0u;
    k.alphaTestEnable     = (p.flags & 4u) != 0u;
    k.pointSpriteEnable   = (p.flags & 8u) != 0u;
    k.samplerDepthCmpMask = p.samplerDepthCmpMask;
    k.samplerNullMask     = p.samplerNullMask;
    k.samplerProjMask     = p.samplerProjMask;
    k.samplerGatherMask   = p.samplerGatherMask;
    return k;
}

MtlPipelineCache::MtlPipelineCache(id<MTLDevice> device) : device_(device) {
    const char *envCache = getenv("ME_PC_SHADER_CACHE");
    std::string dir = (envCache && *envCache) ? envCache : "work/metal_shader_cache";
    mkdir("work", 0755);
    mkdir(dir.c_str(), 0755);
    std::string suffix = pipelineCacheModeSuffix();
    archivePath_ = dir + "/pipelines_v11" + suffix + ".metallib";
    manifestPath_ = dir + "/pso_manifest_v10" + suffix + ".bin";
    if (const char *ae = getenv("ME_PC_MAX_ANISO")) {
        long v = std::strtol(ae, nullptr, 10);
        if (v >= 1 && v <= 16) maxAnisoCap_ = uint8_t(v);
    }
    dispatch_queue_attr_t flushAttr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0);
    archiveFlushQueue_ = dispatch_queue_create("com.runnermetal.archiveflush", flushAttr);

    const char *baEnv = getenv("ME_PC_BINARY_ARCHIVE");
    if (!(baEnv && *baEnv == '0')) {
        MTLBinaryArchiveDescriptor *ad = [[MTLBinaryArchiveDescriptor alloc] init];
        struct stat st = {};
        if (stat(archivePath_.c_str(), &st) == 0 && st.st_size > 0) {
            ad.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:archivePath_.c_str()]];
        }
        NSError *err = nil;
        binaryArchive_ = [device_ newBinaryArchiveWithDescriptor:ad error:&err];
        if (!binaryArchive_ && ad.url != nil) {
            /* Corrupt or older-OS archive: create a fresh archive */
            ad.url = nil;
            binaryArchive_ = [device_ newBinaryArchiveWithDescriptor:ad error:&err];
        }
    }

    loadManifest();
    initFallbacksAndHelpers();
}

MtlPipelineCache::~MtlPipelineCache() {
    if (archiveFlushQueue_) {
        dispatch_sync(archiveFlushQueue_, ^{});
        archiveFlushQueue_ = nullptr;
    }

    std::vector<MTLRenderPipelineDescriptor *> remDescs;
    {
        std::lock_guard<std::mutex> plk(pendingArchiveMu_);
        remDescs.swap(pendingArchiveDescs_);
        pendingArchiveCount_.store(0u, std::memory_order_relaxed);
    }

    if (binaryArchive_) {
        std::lock_guard<std::mutex> lk(archiveMu_);
        for (MTLRenderPipelineDescriptor *d : remDescs) {
            NSError *aerr = nil;
            if ([binaryArchive_ addRenderPipelineFunctionsWithDescriptor:d error:&aerr]) {
                archiveAddedCount_++;
            }
        }
        if (archiveAddedCount_ > archiveFlushedCount_ && !archivePath_.empty()) {
            uint64_t seq = gPsoTmpSeq.fetch_add(1u, std::memory_order_relaxed);
            std::string tmpPath = archivePath_ + ".tmp." + std::to_string(seq) + ".metallib";
            NSURL *tmpUrl = [NSURL fileURLWithPath:[NSString stringWithUTF8String:tmpPath.c_str()]];
            NSError *err = nil;
            if ([binaryArchive_ serializeToURL:tmpUrl error:&err]) {
                std::rename(tmpPath.c_str(), archivePath_.c_str());
                archiveFlushedCount_ = archiveAddedCount_;
            } else {
                std::remove(tmpPath.c_str());
            }
        }
    }

    if (manifestDirty_.exchange(false, std::memory_order_acq_rel) && !manifestPath_.empty()) {
        std::vector<MtlPsoManifestRecord> recordsCopy;
        {
            std::lock_guard<std::mutex> mlk(manifestMu_);
            recordsCopy = manifestRecords_;
        }
        writeManifestFile(manifestPath_, recordsCopy);
    }

    {
        std::lock_guard<std::mutex> clk(psoCacheMu_);
        psoCache_.clear();
    }
    dsCache_.clear();
    sampCache_.clear();
    blitPsoCache_.clear();
    clearPsoCache_.clear();
}

void MtlPipelineCache::loadManifestFromFile(const std::string &path) {
    if (path.empty()) return;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.good()) return;
    MtlPsoManifestHeader hdr = {};
    if (!ifs.read(reinterpret_cast<char *>(&hdr), sizeof(hdr))) return;
    if (hdr.magic != 0x4D505331u || hdr.version != 1u ||
        hdr.recordSize != sizeof(MtlPsoManifestRecord) || hdr.count > 200000u) {
        return;
    }
    if (hdr.count == 0u) return;
    std::vector<MtlPsoManifestRecord> recs(hdr.count);
    if (!ifs.read(reinterpret_cast<char *>(recs.data()),
                  std::streamsize(size_t(hdr.count) * sizeof(MtlPsoManifestRecord)))) {
        return;
    }
    std::lock_guard<std::mutex> lk(manifestMu_);
    manifestRecords_.reserve(manifestRecords_.size() + recs.size());
    manifestSeen_.reserve(manifestRecords_.size() + recs.size());
    for (const auto &r : recs) {
        if (manifestSeen_.insert({r.mapHash, r.psoKey}).second) {
            size_t idx = manifestRecords_.size();
            manifestRecords_.push_back(r);
            if (r.psBytecodeHash != 0) {
                manifestByPs_[r.psBytecodeHash].push_back(idx);
            }
        }
    }
}

void MtlPipelineCache::loadManifest() {
    loadManifestFromFile(manifestPath_);
    if (const char *envManifest = getenv("ME_PC_SHIPPED_MANIFEST"); envManifest && *envManifest) {
        loadManifestFromFile(envManifest);
    }
    char exeBuf[1024] = {};
    uint32_t sz = sizeof(exeBuf);
    if (_NSGetExecutablePath(exeBuf, &sz) == 0) {
        char realExe[1024] = {};
        const char *p = realpath(exeBuf, realExe) ? realExe : exeBuf;
        std::string s(p);
        size_t pos = s.rfind("/MacOS/");
        if (pos != std::string::npos) {
            std::string resPath = s.substr(0, pos) + "/Resources/pso_manifest_v10" + pipelineCacheModeSuffix() + ".bin";
            loadManifestFromFile(resPath);
        }
    }
}

bool MtlPipelineCache::writeManifestFile(const std::string &path, const std::vector<MtlPsoManifestRecord> &records) {
    if (path.empty()) return false;
    uint64_t seq = gPsoTmpSeq.fetch_add(1u, std::memory_order_relaxed);
    std::string tmpPath = path + ".tmp." + std::to_string(seq);
    MtlPsoManifestHeader hdr = {};
    hdr.count = uint32_t(records.size());
    {
        std::ofstream ofs(tmpPath, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) return false;
        ofs.write(reinterpret_cast<const char *>(&hdr), sizeof(hdr));
        if (!records.empty()) {
            ofs.write(reinterpret_cast<const char *>(records.data()),
                      std::streamsize(records.size() * sizeof(MtlPsoManifestRecord)));
        }
        ofs.flush();
        if (!ofs.good()) {
            ofs.close();
            std::remove(tmpPath.c_str());
            return false;
        }
    }
    if (std::rename(tmpPath.c_str(), path.c_str()) != 0) {
        std::remove(tmpPath.c_str());
        return false;
    }
    return true;
}

void MtlPipelineCache::seedCompilerFromManifest(MtlShaderCompiler &compiler) {
    std::vector<MtlPsoManifestRecord> recs;
    {
        std::lock_guard<std::mutex> lk(manifestMu_);
        recs = manifestRecords_;
    }
    for (const auto &r : recs) {
        if (r.psBytecodeHash != 0) {
            compiler.seedKnownPsVariant(r.psBytecodeHash, r.vsBytecodeHash, unpackPsVariantKey(r.psKey));
        }
    }
}

void MtlPipelineCache::recordManifestEntry(
    uint64_t mapHash,
    uint64_t vsBytecodeHash,
    uint64_t psBytecodeHash,
    const MtlShaderVariantKey &psKey,
    const MtlRenderPipelineKey &psoKey,
    const MtlVertexDeclaration9 *decl,
    const MtlStreamBindingInfo streams[16]) {
    if (mapHash == 0 || vsBytecodeHash == 0) return;

    MtlPsoManifestRecord rec = {};
    rec.mapHash        = mapHash;
    rec.vsBytecodeHash = vsBytecodeHash;
    rec.psBytecodeHash = psBytecodeHash;
    rec.psKey          = packPsVariantKey(psKey);
    rec.psoKey         = psoKey;
    if (decl) {
        size_t cnt = std::min<size_t>(decl->elements().size(), 24u);
        rec.elemCount = uint8_t(cnt);
        for (size_t i = 0; i < cnt; i++) rec.elems[i] = decl->elements()[i];
    }
    for (uint32_t s = 0; s < 16u; s++) {
        if (streams[s].bound && streams[s].stride > 0) {
            rec.streamBoundMask |= uint16_t(1u << s);
            rec.streamStrides[s] = streams[s].stride;
            rec.streamFreqs[s]   = streams[s].freqSetting;
        }
    }

    {
        std::lock_guard<std::mutex> lk(manifestMu_);
        if (!manifestSeen_.insert({mapHash, psoKey}).second) return;
        size_t idx = manifestRecords_.size();
        manifestRecords_.push_back(rec);
        if (rec.psBytecodeHash != 0) {
            manifestByPs_[rec.psBytecodeHash].push_back(idx);
        }
    }
    uint64_t nowMs = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    lastManifestUpdateMs_.store(nowMs, std::memory_order_relaxed);
    manifestDirty_.store(true, std::memory_order_release);
}

std::vector<MtlPsoManifestRecord> MtlPipelineCache::getManifestRecordsForMap(uint64_t mapHash) {
    std::vector<MtlPsoManifestRecord> out;
    if (mapHash == 0) return out;
    std::lock_guard<std::mutex> lk(manifestMu_);
    for (const auto &r : manifestRecords_) {
        if (r.mapHash == mapHash) out.push_back(r);
    }
    return out;
}

std::vector<MtlPsoManifestRecord> MtlPipelineCache::getAllManifestRecordsExceptMap(uint64_t excludeMapHash) {
    std::vector<MtlPsoManifestRecord> out;
    std::lock_guard<std::mutex> lk(manifestMu_);
    std::unordered_set<MtlRenderPipelineKey, MtlRenderPipelineKeyHash> seen;
    seen.reserve(manifestRecords_.size());
    for (const auto &r : manifestRecords_) {
        if (r.mapHash != excludeMapHash && seen.insert(r.psoKey).second) {
            out.push_back(r);
        }
    }
    return out;
}

void MtlPipelineCache::warmManifestPsosForVariant(
    uint64_t psBytecodeHash,
    uint64_t vsBytecodeHash,
    const MtlShaderVariantKey &psKey,
    const MtlParsedShader &vs,
    const MtlCompiledShader &ps) {
    if (psBytecodeHash == 0 || vsBytecodeHash == 0) return;

    std::vector<MtlPsoManifestRecord> matches;
    {
        std::lock_guard<std::mutex> lk(manifestMu_);
        auto it = manifestByPs_.find(psBytecodeHash);
        if (it == manifestByPs_.end()) return;
        for (size_t idx : it->second) {
            if (idx >= manifestRecords_.size()) continue;
            const MtlPsoManifestRecord &r = manifestRecords_[idx];
            if (r.vsBytecodeHash == vsBytecodeHash && unpackPsVariantKey(r.psKey) == psKey) {
                matches.push_back(r);
                if (matches.size() >= 16u) break;
            }
        }
    }

    for (const auto &rec : matches) {
        MtlStreamBindingInfo streams[16] = {};
        for (uint32_t s = 0; s < 16u; s++) {
            streams[s].bound       = (rec.streamBoundMask & uint16_t(1u << s)) != 0u;
            streams[s].stride      = rec.streamStrides[s];
            streams[s].freqSetting = rec.streamFreqs[s];
        }
        warmRenderPipeline(rec.psoKey, vs, &ps, rec.elems, rec.elemCount, streams);
    }
}

void MtlPipelineCache::initFallbacksAndHelpers() {
    /* 1x1 opaque black fallback textures (D3D9 unbound sampler returns (0,0,0,1)) */
    const uint8_t blackPixel[4] = { 0, 0, 0, 255 };

    {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                     width:1 height:1 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModeShared;
        nullTex2D_ = [device_ newTextureWithDescriptor:d];
        [nullTex2D_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 withBytes:blackPixel bytesPerRow:4];
    }
    {
        MTLTextureDescriptor *d = [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                        size:1 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModeShared;
        nullTexCube_ = [device_ newTextureWithDescriptor:d];
        for (NSUInteger face = 0; face < 6; face++) {
            [nullTexCube_ replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0 slice:face
                              withBytes:blackPixel bytesPerRow:4 bytesPerImage:4];
        }
    }
    {
        MTLTextureDescriptor *d = [[MTLTextureDescriptor alloc] init];
        d.textureType = MTLTextureType3D;
        d.pixelFormat = MTLPixelFormatRGBA8Unorm;
        d.width = 1; d.height = 1; d.depth = 1;
        d.usage = MTLTextureUsageShaderRead;
        d.storageMode = MTLStorageModeShared;
        nullTex3D_ = [device_ newTextureWithDescriptor:d];
        [nullTex3D_ replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:0
                        withBytes:blackPixel bytesPerRow:4 bytesPerImage:4];
    }

    zeroAttrBuffer_ = [device_ newBufferWithLength:64 options:MTLResourceStorageModeShared];
    if (zeroAttrBuffer_) std::memset([zeroAttrBuffer_ contents], 0, 64);

    {
        MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
        sd.minFilter = MTLSamplerMinMagFilterNearest;
        sd.magFilter = MTLSamplerMinMagFilterNearest;
        sd.mipFilter = MTLSamplerMipFilterNotMipmapped;
        sd.sAddressMode = MTLSamplerAddressModeRepeat;
        sd.tAddressMode = MTLSamplerAddressModeRepeat;
        sd.rAddressMode = MTLSamplerAddressModeRepeat;
        defaultSampler_ = [device_ newSamplerStateWithDescriptor:sd];

        sd.sAddressMode = MTLSamplerAddressModeClampToEdge;
        sd.tAddressMode = MTLSamplerAddressModeClampToEdge;
        sd.rAddressMode = MTLSamplerAddressModeClampToEdge;
        pointClampSampler_ = [device_ newSamplerStateWithDescriptor:sd];

        sd.minFilter = MTLSamplerMinMagFilterLinear;
        sd.magFilter = MTLSamplerMinMagFilterLinear;
        linearClampSampler_ = [device_ newSamplerStateWithDescriptor:sd];
    }

    NSError *err = nil;
    helperLib_ = [device_ newLibraryWithSource:[NSString stringWithUTF8String:kHelperShadersMsl]
                                       options:nil error:&err];
    if (!helperLib_) {
        fprintf(stderr, "RunnerMetal: helper MSL compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return;
    }
    blitVs_     = [helperLib_ newFunctionWithName:@"mtl_blit_vs"];
    blitPs_     = [helperLib_ newFunctionWithName:@"mtl_blit_ps"];
    blitPsHalf_ = [helperLib_ newFunctionWithName:@"mtl_blit_ps_half"];
    clearVs_    = [helperLib_ newFunctionWithName:@"mtl_clear_vs"];
    clearPs_    = [helperLib_ newFunctionWithName:@"mtl_clear_ps"];
}

id<MTLTexture> MtlPipelineCache::nullTexture(MtlSamplerTexKind kind) const {
    switch (kind) {
        case MtlSamplerTexKind::TexCube: return nullTexCube_;
        case MtlSamplerTexKind::Tex3D:   return nullTex3D_;
        default:                         return nullTex2D_;
    }
}

static MTLColorWriteMask mapColorWriteMask(uint8_t d3dMask) {
    MTLColorWriteMask m = MTLColorWriteMaskNone;
    if (d3dMask & 1u) m |= MTLColorWriteMaskRed;
    if (d3dMask & 2u) m |= MTLColorWriteMaskGreen;
    if (d3dMask & 4u) m |= MTLColorWriteMaskBlue;
    if (d3dMask & 8u) m |= MTLColorWriteMaskAlpha;
    return m;
}

MTLRenderPipelineDescriptor *MtlPipelineCache::buildRenderPipelineDescriptor(
    const MtlRenderPipelineKey &key,
    const MtlParsedShader &vs,
    const MtlCompiledShader *ps,
    const D3DMtlVertexElement *elems,
    size_t elemCount,
    const MtlStreamBindingInfo streams[16]) const {

    if (!vs.defaultCompiled.function) return nil;

    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = vs.defaultCompiled.function;
    bool hasColorAttachment = (key.colorFormats[0] | key.colorFormats[1] |
                               key.colorFormats[2] | key.colorFormats[3]) != 0;
    bool skipEmptyPs = ps && !hasColorAttachment && !ps->writesDepth && !ps->usesDiscard;
    desc.fragmentFunction = (ps && ps->function && !skipEmptyPs) ? ps->function : nil;
    desc.rasterSampleCount = std::max<uint8_t>(key.sampleCount, 1);
    desc.alphaToCoverageEnabled = (key.alphaToCoverage != 0);

    /* Vertex descriptor matched by D3D9 (Usage, UsageIndex) semantic */
    MTLVertexDescriptor *vd = [[MTLVertexDescriptor alloc] init];
    bool usedStream[16] = {};
    bool usedZero = false;

    for (const auto &inSem : vs.vsInputs) {
        const D3DMtlVertexElement *matched = nullptr;
        if (elems && elemCount > 0u) {
            for (size_t i = 0; i < elemCount; i++) {
                const D3DMtlVertexElement &e = elems[i];
                if (e.Usage == inSem.usage && e.UsageIndex == inSem.usageIndex) {
                    matched = &e;
                    break;
                }
            }
        }

        if (matched && matched->Stream < 16 && streams[matched->Stream].bound && streams[matched->Stream].stride > 0) {
            vd.attributes[inSem.location].format      = mapD3dDeclType(matched->Type);
            vd.attributes[inSem.location].offset      = matched->Offset;
            vd.attributes[inSem.location].bufferIndex = matched->Stream;
            usedStream[matched->Stream] = true;
        } else {
            vd.attributes[inSem.location].format      = MTLVertexFormatFloat4;
            vd.attributes[inSem.location].offset      = 0;
            vd.attributes[inSem.location].bufferIndex = kVsBufZeroAttr;
            usedZero = true;
        }
    }

    for (uint32_t s = 0; s < 16; s++) {
        if (!usedStream[s]) continue;
        vd.layouts[s].stride = streams[s].stride;
        if ((streams[s].freqSetting & D3DM_STREAMSOURCE_INSTANCEDATA) != 0) {
            vd.layouts[s].stepFunction = MTLVertexStepFunctionPerInstance;
            vd.layouts[s].stepRate     = std::max(1u, streams[s].freqSetting & 0x3FFFFFFFu);
        } else {
            vd.layouts[s].stepFunction = MTLVertexStepFunctionPerVertex;
            vd.layouts[s].stepRate     = 1;
        }
    }
    if (usedZero) {
        vd.layouts[kVsBufZeroAttr].stride       = 16;
        vd.layouts[kVsBufZeroAttr].stepFunction = MTLVertexStepFunctionConstant;
        vd.layouts[kVsBufZeroAttr].stepRate     = 0;
    }
    desc.vertexDescriptor = vd;

    /* Color attachments 0..3 */
    for (uint32_t i = 0; i < 4; i++) {
        MTLPixelFormat cf = static_cast<MTLPixelFormat>(key.colorFormats[i]);
        desc.colorAttachments[i].pixelFormat = cf;
        if (cf == MTLPixelFormatInvalid) continue;

        desc.colorAttachments[i].writeMask = mapColorWriteMask(key.colorWriteMasks[i]);

        if (key.blendEnable) {
            bool hasAlpha = (key.colorHasAlphaMask & (1u << i)) != 0;
            uint32_t srcB = key.srcBlend;
            uint32_t dstB = key.dstBlend;
            if (srcB == D3DM_BLEND_BOTHSRCALPHA) {
                srcB = D3DM_BLEND_SRCALPHA;
                dstB = D3DM_BLEND_INVSRCALPHA;
            } else if (srcB == D3DM_BLEND_BOTHINVSRCALPHA) {
                srcB = D3DM_BLEND_INVSRCALPHA;
                dstB = D3DM_BLEND_SRCALPHA;
            }

            MTLBlendFactor srcRgb = mapD3dBlendFactor(srcB, hasAlpha);
            MTLBlendFactor dstRgb = mapD3dBlendFactor(dstB, hasAlpha);
            MTLBlendOperation opRgb = mapD3dBlendOp(key.blendOp);

            MTLBlendFactor srcA = key.sepAlphaEnable ? mapD3dBlendFactor(key.srcBlendAlpha, hasAlpha) : srcRgb;
            MTLBlendFactor dstA = key.sepAlphaEnable ? mapD3dBlendFactor(key.dstBlendAlpha, hasAlpha) : dstRgb;
            MTLBlendOperation opA = key.sepAlphaEnable ? mapD3dBlendOp(key.blendOpAlpha) : opRgb;

            desc.colorAttachments[i].blendingEnabled             = YES;
            desc.colorAttachments[i].sourceRGBBlendFactor        = srcRgb;
            desc.colorAttachments[i].destinationRGBBlendFactor   = dstRgb;
            desc.colorAttachments[i].rgbBlendOperation           = opRgb;
            desc.colorAttachments[i].sourceAlphaBlendFactor      = srcA;
            desc.colorAttachments[i].destinationAlphaBlendFactor = dstA;
            desc.colorAttachments[i].alphaBlendOperation         = opA;
        }
    }

    desc.depthAttachmentPixelFormat   = static_cast<MTLPixelFormat>(key.depthFormat);
    desc.stencilAttachmentPixelFormat = static_cast<MTLPixelFormat>(key.stencilFormat);
    return desc;
}

id<MTLRenderPipelineState> MtlPipelineCache::getOrCreateRenderPipeline(
    const MtlRenderPipelineKey &key,
    const MtlParsedShader &vs,
    const MtlCompiledShader *ps,
    const MtlVertexDeclaration9 *decl,
    const MtlStreamBindingInfo streams[16],
    bool *outPsoCacheHit,
    bool *outWarmPsoHit,
    uint64_t recordMapHash,
    bool *outNeedManifestRecord) {

    if (outNeedManifestRecord) *outNeedManifestRecord = false;
    {
        std::lock_guard<std::mutex> clk(psoCacheMu_);
        auto it = psoCache_.find(key);
        if (it != psoCache_.end()) {
            if (outPsoCacheHit) *outPsoCacheHit = true;
            if (it->second.warmUnconsumed) {
                it->second.warmUnconsumed = false;
                if (outWarmPsoHit) *outWarmPsoHit = true;
            } else if (outWarmPsoHit) {
                *outWarmPsoHit = false;
            }
            if (recordMapHash != 0 && it->second.recordedMapHash != recordMapHash) {
                it->second.recordedMapHash = recordMapHash;
                if (outNeedManifestRecord) *outNeedManifestRecord = true;
            }
            return it->second.pso;
        }
    }
    if (outPsoCacheHit) *outPsoCacheHit = false;
    if (outWarmPsoHit) *outWarmPsoHit = false;

    const D3DMtlVertexElement *elems = decl ? decl->elements().data() : nullptr;
    size_t elemCount = decl ? decl->elements().size() : 0u;
    MTLRenderPipelineDescriptor *desc = buildRenderPipelineDescriptor(key, vs, ps, elems, elemCount, streams);
    if (!desc) return nil;

    static bool logStalls = (getenv("ME_PC_LOG_STALLS") && *getenv("ME_PC_LOG_STALLS") == '1');
    auto t0 = logStalls ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    double lockMs = 0.0, psoMs = 0.0;

    id<MTLRenderPipelineState> pso = nil;
    NSError *err = nil;
    bool archiveHit = false;
    if (binaryArchive_) {
        std::unique_lock<std::mutex> alk(archiveMu_, std::try_to_lock);
        if (alk.owns_lock()) {
            if (logStalls) {
                lockMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            }
            desc.binaryArchives = @[binaryArchive_];
            pso = [device_ newRenderPipelineStateWithDescriptor:desc
                                                        options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                     reflection:nil
                                                          error:nil];
            if (pso) {
                archiveHit = true;
            }
        }
    }
    if (!pso) {
        desc.binaryArchives = nil;
        {
            struct PsoStallGuard {
                bool armed = (host_stall_hook != nullptr);
                PsoStallGuard() { if (armed) host_stall_hook(1); }
                ~PsoStallGuard() { if (armed) host_stall_hook(0); }
            } psoStallGuard;
            pso = [device_ newRenderPipelineStateWithDescriptor:desc error:&err];
        }
        if (pso && binaryArchive_) {
            MTLRenderPipelineDescriptor *descCopy = [desc copy];
            descCopy.binaryArchives = nil;
            if (descCopy) {
                std::lock_guard<std::mutex> plk(pendingArchiveMu_);
                pendingArchiveDescs_.push_back(descCopy);
                pendingArchiveCount_.fetch_add(1u, std::memory_order_relaxed);
            }
            uint64_t nowMs = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            lastArchiveAddMs_.store(nowMs, std::memory_order_relaxed);
        }
    }
    if (logStalls) {
        psoMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() - lockMs;
        if ((lockMs + psoMs) > 2.0) {
            fprintf(stderr, "STALL_LOG: getOrCreateRenderPipeline %s lock=%.2f ms pso=%.2f ms\n",
                    archiveHit ? "archiveHit" : "miss", lockMs, psoMs);
        }
    }

    if (!pso) {
        fprintf(stderr, "RunnerMetal: PSO creation failed: %s (color0=%u ds=%u/%u)\n",
                err ? [[err localizedDescription] UTF8String] : "unknown",
                key.colorFormats[0], key.depthFormat, key.stencilFormat);
        return nil;
    }

    std::lock_guard<std::mutex> clk(psoCacheMu_);
    auto [ins, ok] = psoCache_.emplace(key, CachedPsoEntry{pso, false, recordMapHash});
    if (!ok && recordMapHash != 0 && ins->second.recordedMapHash != recordMapHash) {
        ins->second.recordedMapHash = recordMapHash;
        if (outNeedManifestRecord) *outNeedManifestRecord = true;
    } else if (ok && recordMapHash != 0) {
        if (outNeedManifestRecord) *outNeedManifestRecord = true;
    }
    return ins->second.pso;
}

bool MtlPipelineCache::warmRenderPipeline(
    const MtlRenderPipelineKey &key,
    const MtlParsedShader &vs,
    const MtlCompiledShader *ps,
    const D3DMtlVertexElement *elems,
    size_t elemCount,
    const MtlStreamBindingInfo streams[16]) {

    {
        std::lock_guard<std::mutex> clk(psoCacheMu_);
        if (psoCache_.find(key) != psoCache_.end()) return false;
    }

    MTLRenderPipelineDescriptor *desc = buildRenderPipelineDescriptor(key, vs, ps, elems, elemCount, streams);
    if (!desc) return false;

    id<MTLRenderPipelineState> pso = nil;
    NSError *err = nil;
    if (binaryArchive_) {
        std::unique_lock<std::mutex> alk(archiveMu_, std::try_to_lock);
        if (alk.owns_lock()) {
            desc.binaryArchives = @[binaryArchive_];
            pso = [device_ newRenderPipelineStateWithDescriptor:desc
                                                        options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                     reflection:nil
                                                          error:nil];
        }
    }
    if (!pso) {
        desc.binaryArchives = nil;
        pso = [device_ newRenderPipelineStateWithDescriptor:desc error:&err];
        if (pso && binaryArchive_) {
            MTLRenderPipelineDescriptor *descCopy = [desc copy];
            descCopy.binaryArchives = nil;
            if (descCopy) {
                std::lock_guard<std::mutex> plk(pendingArchiveMu_);
                pendingArchiveDescs_.push_back(descCopy);
                pendingArchiveCount_.fetch_add(1u, std::memory_order_relaxed);
            }
            uint64_t nowMs = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
            lastArchiveAddMs_.store(nowMs, std::memory_order_relaxed);
        }
    }
    if (!pso) return false;

    std::lock_guard<std::mutex> clk(psoCacheMu_);
    auto [ins, ok] = psoCache_.emplace(key, CachedPsoEntry{pso, true});
    return ok;
}

void MtlPipelineCache::maybeFlushArchive() {
    if (!archiveFlushQueue_) return;
    uint32_t pendingDescs = pendingArchiveCount_.load(std::memory_order_relaxed);
    bool dirtyManifest = manifestDirty_.load(std::memory_order_relaxed);
    uint32_t pendingArchive = (binaryArchive_ && !archivePath_.empty())
        ? (archiveAddedCount_ - archiveFlushedCount_) : 0u;
    if (pendingDescs == 0u && pendingArchive == 0u && !dirtyManifest) return;

    uint64_t nowMs = uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
    bool trigger = false;
    if (pendingDescs >= 64u) trigger = true;
    else if (pendingDescs > 0u && nowMs >= lastArchiveAddMs_.load(std::memory_order_relaxed) + 1500ull) trigger = true;
    else if (pendingArchive >= 128u || (pendingArchive > 0u && nowMs >= lastArchiveAddMs_.load(std::memory_order_relaxed) + 5000ull)) trigger = true;
    else if (dirtyManifest && nowMs >= lastManifestUpdateMs_.load(std::memory_order_relaxed) + 2000ull) trigger = true;
    if (!trigger) return;

    if (archiveFlushPending_.exchange(true, std::memory_order_acq_rel)) return;

    dispatch_async(archiveFlushQueue_, ^{
        @autoreleasepool {
            std::vector<MTLRenderPipelineDescriptor *> localDescs;
            {
                std::lock_guard<std::mutex> plk(pendingArchiveMu_);
                localDescs.swap(pendingArchiveDescs_);
                pendingArchiveCount_.store(0u, std::memory_order_relaxed);
            }

            if (binaryArchive_ && (!localDescs.empty() || archiveAddedCount_ > archiveFlushedCount_)) {
                std::lock_guard<std::mutex> alk(archiveMu_);
                for (MTLRenderPipelineDescriptor *d : localDescs) {
                    NSError *aerr = nil;
                    if ([binaryArchive_ addRenderPipelineFunctionsWithDescriptor:d error:&aerr]) {
                        archiveAddedCount_++;
                    }
                }
                if (!archivePath_.empty() && archiveAddedCount_ > archiveFlushedCount_) {
                    archiveFlushedCount_ = archiveAddedCount_;
                    uint64_t seq = gPsoTmpSeq.fetch_add(1u, std::memory_order_relaxed);
                    std::string tmpPath = archivePath_ + ".tmp." + std::to_string(seq) + ".metallib";
                    NSURL *tmpUrl = [NSURL fileURLWithPath:[NSString stringWithUTF8String:tmpPath.c_str()]];
                    NSError *err = nil;
                    if ([binaryArchive_ serializeToURL:tmpUrl error:&err]) {
                        std::rename(tmpPath.c_str(), archivePath_.c_str());
                    } else {
                        std::remove(tmpPath.c_str());
                    }
                }
            }

            if (manifestDirty_.exchange(false, std::memory_order_acq_rel) && !manifestPath_.empty()) {
                std::vector<MtlPsoManifestRecord> recordsCopy;
                {
                    std::lock_guard<std::mutex> mlk(manifestMu_);
                    recordsCopy = manifestRecords_;
                }
                writeManifestFile(manifestPath_, recordsCopy);
            }
        }
        archiveFlushPending_.store(false, std::memory_order_release);
    });
}

id<MTLDepthStencilState> MtlPipelineCache::getOrCreateDepthStencilState(const MtlDepthStencilKey &key) {
    auto it = dsCache_.find(key);
    if (it != dsCache_.end()) return it->second;

    MTLDepthStencilDescriptor *desc = [[MTLDepthStencilDescriptor alloc] init];
    if (key.hasDepth && key.zEnable != 0) {
        desc.depthCompareFunction = mapD3dCmpFunc(key.zFunc);
        desc.depthWriteEnabled    = (key.zWrite != 0);
    } else {
        desc.depthCompareFunction = MTLCompareFunctionAlways;
        desc.depthWriteEnabled    = NO;
    }

    if (key.hasStencil && key.stencilEnable != 0) {
        MTLStencilDescriptor *front = [[MTLStencilDescriptor alloc] init];
        front.stencilCompareFunction    = mapD3dCmpFunc(key.frontFunc);
        front.stencilFailureOperation   = mapD3dStencilOp(key.frontFail);
        front.depthFailureOperation     = mapD3dStencilOp(key.frontZFail);
        front.depthStencilPassOperation = mapD3dStencilOp(key.frontPass);
        front.readMask                  = key.readMask;
        front.writeMask                 = key.writeMask;
        desc.frontFaceStencil = front;

        if (key.twoSided) {
            MTLStencilDescriptor *back = [[MTLStencilDescriptor alloc] init];
            back.stencilCompareFunction    = mapD3dCmpFunc(key.backFunc);
            back.stencilFailureOperation   = mapD3dStencilOp(key.backFail);
            back.depthFailureOperation     = mapD3dStencilOp(key.backZFail);
            back.depthStencilPassOperation = mapD3dStencilOp(key.backPass);
            back.readMask                  = key.readMask;
            back.writeMask                 = key.writeMask;
            desc.backFaceStencil = back;
        } else {
            desc.backFaceStencil = front;
        }
    }

    id<MTLDepthStencilState> state = [device_ newDepthStencilStateWithDescriptor:desc];
    dsCache_.emplace(key, state);
    return state;
}

id<MTLSamplerState> MtlPipelineCache::getOrCreateSamplerState(const MtlSamplerKey &key) {
    auto it = sampCache_.find(key);
    if (it != sampCache_.end()) return it->second;

    MTLSamplerDescriptor *sd = [[MTLSamplerDescriptor alloc] init];
    sd.sAddressMode = mapD3dAddressMode(key.addrU);
    sd.tAddressMode = mapD3dAddressMode(key.addrV);
    sd.rAddressMode = mapD3dAddressMode(key.addrW);

    sd.minFilter = (key.minFilter == D3DM_TEXF_LINEAR || key.minFilter == D3DM_TEXF_ANISOTROPIC)
        ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    sd.magFilter = (key.magFilter == D3DM_TEXF_LINEAR || key.magFilter == D3DM_TEXF_ANISOTROPIC)
        ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;

    switch (key.mipFilter) {
        case D3DM_TEXF_POINT:       sd.mipFilter = MTLSamplerMipFilterNearest; break;
        case D3DM_TEXF_LINEAR:
        case D3DM_TEXF_ANISOTROPIC: sd.mipFilter = MTLSamplerMipFilterLinear;  break;
        default:                    sd.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
    }

    if (key.minFilter == D3DM_TEXF_ANISOTROPIC || key.magFilter == D3DM_TEXF_ANISOTROPIC) {
        sd.maxAnisotropy = std::clamp<NSUInteger>(key.maxAniso, 1, maxAnisoCap_);
    } else {
        sd.maxAnisotropy = 1;
    }

    sd.lodMinClamp = float(key.maxMipLevel);
    if (key.depthCompare) {
        sd.minFilter = MTLSamplerMinMagFilterLinear;
        sd.magFilter = MTLSamplerMinMagFilterLinear;
        sd.maxAnisotropy = 1;
        sd.compareFunction = MTLCompareFunctionLessEqual;
    } else {
        sd.compareFunction = MTLCompareFunctionNever;
    }

    uint8_t a = uint8_t((key.borderColor >> 24) & 0xFFu);
    uint8_t r = uint8_t((key.borderColor >> 16) & 0xFFu);
    uint8_t g = uint8_t((key.borderColor >>  8) & 0xFFu);
    uint8_t b = uint8_t((key.borderColor >>  0) & 0xFFu);
    if (a < 128) {
        sd.borderColor = MTLSamplerBorderColorTransparentBlack;
    } else if (uint32_t(r) + g + b > 384u) {
        sd.borderColor = MTLSamplerBorderColorOpaqueWhite;
    } else {
        sd.borderColor = MTLSamplerBorderColorOpaqueBlack;
    }

    id<MTLSamplerState> smp = [device_ newSamplerStateWithDescriptor:sd];
    sampCache_.emplace(key, smp);
    return smp;
}

static bool isLowPrecisionBlitFormat(MTLPixelFormat fmt) {
    switch (fmt) {
        case MTLPixelFormatBGRA8Unorm:
        case MTLPixelFormatBGRA8Unorm_sRGB:
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGR10A2Unorm:
        case MTLPixelFormatRGB10A2Unorm:
            return true;
        default:
            return false;
    }
}

id<MTLRenderPipelineState> MtlPipelineCache::getBlitPipeline(MTLPixelFormat dstFmt) {
    uint32_t k = uint32_t(dstFmt);
    auto it = blitPsoCache_.find(k);
    if (it != blitPsoCache_.end()) return it->second;

    if (!blitVs_ || !blitPs_) return nil;
    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = blitVs_;
    desc.fragmentFunction = (isLowPrecisionBlitFormat(dstFmt) && blitPsHalf_) ? blitPsHalf_ : blitPs_;
    desc.colorAttachments[0].pixelFormat = dstFmt;
    NSError *err = nil;
    id<MTLRenderPipelineState> pso = [device_ newRenderPipelineStateWithDescriptor:desc error:&err];
    if (!pso) {
        fprintf(stderr, "RunnerMetal: blit PSO failed for fmt %lu: %s\n",
                (unsigned long)dstFmt, err ? [[err localizedDescription] UTF8String] : "unknown");
        return nil;
    }
    blitPsoCache_.emplace(k, pso);
    return pso;
}

id<MTLRenderPipelineState> MtlPipelineCache::getClearPipeline(MTLPixelFormat dstFmt, MTLPixelFormat dsFmt,
                                                              uint8_t colorWriteMask) {
    uint8_t cwMask = (dstFmt != MTLPixelFormatInvalid) ? (colorWriteMask & 0xFu) : 0u;
    uint64_t k = (uint64_t(uint32_t(dstFmt) & 0xFFFFu) << 20) |
                 (uint64_t(uint32_t(dsFmt)  & 0xFFFFu) << 4)  |
                 uint64_t(cwMask);
    auto it = clearPsoCache_.find(k);
    if (it != clearPsoCache_.end()) return it->second;

    if (!clearVs_ || !clearPs_) return nil;
    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction = clearVs_;
    desc.fragmentFunction = (dstFmt != MTLPixelFormatInvalid && cwMask != 0u) ? clearPs_ : nil;
    desc.colorAttachments[0].pixelFormat = dstFmt;
    if (dstFmt != MTLPixelFormatInvalid) {
        desc.colorAttachments[0].writeMask = mapColorWriteMask(cwMask);
    }
    desc.depthAttachmentPixelFormat = dsFmt;
    desc.stencilAttachmentPixelFormat =
        (dsFmt == MTLPixelFormatDepth32Float_Stencil8) ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatInvalid;
    NSError *err = nil;
    id<MTLRenderPipelineState> pso = [device_ newRenderPipelineStateWithDescriptor:desc error:&err];
    if (!pso) {
        fprintf(stderr, "RunnerMetal: clear PSO failed for fmt %lu/%lu mask=0x%x: %s\n",
                (unsigned long)dstFmt, (unsigned long)dsFmt, cwMask,
                err ? [[err localizedDescription] UTF8String] : "unknown");
        return nil;
    }
    clearPsoCache_.emplace(k, pso);
    return pso;
}

} // namespace mtl_d3d9
