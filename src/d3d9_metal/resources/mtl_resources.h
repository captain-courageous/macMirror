#pragma once

#import <Metal/Metal.h>
#include "d3d9_metal_com.h"
#include "d3d9_metal_enums.h"
#include "mtl_format.h"
#include "mtl_shader_compiler.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace mtl_d3d9 {

class MtlDirect3DDevice9;

/* Triple-buffered MTLStorageModeShared unified-memory ring buffer */
class MtlRingBuffer {
public:
    static constexpr uint32_t kFrameSlots = 3;
    static constexpr size_t   kSlotSize   = 64u * 1024u * 1024u; /* 64 MB per in-flight frame */

    struct Allocation {
        id<MTLBuffer> buffer = nil;
        size_t        offset = 0;
        void         *cpuPtr = nullptr;
    };

    explicit MtlRingBuffer(id<MTLDevice> device);
    ~MtlRingBuffer();

    void beginFrame(uint32_t frameIndex);
    Allocation alloc(size_t size, size_t alignment = 16);

private:
    id<MTLDevice> device_;
    id<MTLBuffer> slots_[kFrameSlots] = {};
    void         *slotPtrs_[kFrameSlots] = {};
    size_t        offsets_[kFrameSlots] = {};
    std::vector<id<MTLBuffer>> overflow_[kFrameSlots];
    uint32_t      curSlot_ = 0;
};

struct PendingTextureUpload {
    id<MTLBuffer>  srcBuffer = nil;
    size_t         srcOffset = 0;
    uint32_t       srcBytesPerRow = 0;
    uint32_t       srcBytesPerImage = 0;
    id<MTLTexture> dstTexture = nil;
    uint32_t       dstSlice = 0;
    uint32_t       dstLevel = 0;
    MTLOrigin      dstOrigin = {0, 0, 0};
    MTLSize        dstSize = {0, 0, 1};
};

/* ---- Vertex & Index Buffers ---- */
struct MtlBufferDynSlot {
    id<MTLBuffer> buffer = nil;
    uint64_t      lastSeq = 0;
};

/* Teardown-safe recycled shared MTLBuffer pool for dynamic/temporary VBs & IBs */
struct MtlSharedBufferPool {
    struct Entry {
        id<MTLBuffer> buffer = nil;
        uint64_t      readySeq = 0;
        size_t        size = 0;
    };
    id<MTLDevice>          device = nil;
    std::atomic<uint64_t> *completedSeq = nullptr;
    std::mutex             mu;
    std::vector<Entry>     entries;
    size_t                 totalBytes = 0;

    MtlSharedBufferPool(id<MTLDevice> dev, std::atomic<uint64_t> *seqPtr)
        : device(dev), completedSeq(seqPtr) {}

    void shutdown() {
        std::lock_guard<std::mutex> lk(mu);
        completedSeq = nullptr;
        entries.clear();
        totalBytes = 0;
        device = nil;
    }

    id<MTLBuffer> acquire(size_t minBytes) {
        if (minBytes < 16u) minBytes = 16u;
        id<MTLDevice> dev = nil;
        {
            std::lock_guard<std::mutex> lk(mu);
            dev = device;
            if (completedSeq) {
                uint64_t doneSeq = completedSeq->load(std::memory_order_acquire);
                size_t bestIdx = SIZE_MAX;
                size_t bestSize = SIZE_MAX;
                for (size_t i = 0; i < entries.size(); i++) {
                    const auto &e = entries[i];
                    if (e.readySeq <= doneSeq && e.size >= minBytes &&
                        e.size <= minBytes * 2u && e.size < bestSize) {
                        bestIdx = i;
                        bestSize = e.size;
                        if (bestSize == minBytes) break;
                    }
                }
                if (bestIdx != SIZE_MAX) {
                    id<MTLBuffer> buf = entries[bestIdx].buffer;
                    totalBytes -= entries[bestIdx].size;
                    entries[bestIdx] = std::move(entries.back());
                    entries.pop_back();
                    return buf;
                }
            }
        }
        return dev ? [dev newBufferWithLength:minBytes options:MTLResourceStorageModeShared] : nil;
    }

    void release(id<MTLBuffer> buf, uint64_t readySeq) {
        if (!buf) return;
        size_t sz = [buf length];
        if (sz < 256u || sz > 4u * 1024u * 1024u) return;
        std::lock_guard<std::mutex> lk(mu);
        if (!completedSeq || entries.size() >= 64u || totalBytes + sz > 32u * 1024u * 1024u) return;
        totalBytes += sz;
        entries.push_back({ buf, readySeq, sz });
    }
};

class MtlVertexBuffer9 final : public MtlComObject<MtlVertexBuffer9, IMtlDirect3DVertexBuffer9> {
public:
    MtlVertexBuffer9(MtlDirect3DDevice9 *dev, uint32_t length, uint32_t usage, uint32_t fvf, uint32_t pool);
    ~MtlVertexBuffer9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT SetPrivateData(const D3DMtlGuid *, const void *, uint32_t, uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetPrivateData(const D3DMtlGuid *, void *, uint32_t *) override { return D3D_ERR_NOTAVAILABLE; }
    MTL_HRESULT FreePrivateData(const D3DMtlGuid *) override { return D3D_S_OK; }
    uint32_t    SetPriority(uint32_t p) override { uint32_t old = priority_; priority_ = p; return old; }
    uint32_t    GetPriority() override { return priority_; }
    void        PreLoad() override {}
    uint32_t    GetType() override { return D3DM_RTYPE_VERTEXBUFFER; }
    MTL_HRESULT Lock(uint32_t offset, uint32_t size, void **ppbData, uint32_t flags) override;
    MTL_HRESULT Unlock() override;
    MTL_HRESULT GetDesc(D3DMtlBufferDesc *pDesc) override;

    id<MTLBuffer> activeBuffer() const { return activeBuffer_; }
    size_t        activeOffset() const { return activeOffset_; }
    uint32_t      byteLength() const { return desc_.Size; }
    void          markUsed(uint64_t cmdSeq);

private:
    MtlDirect3DDevice9 *device_;
    D3DMtlBufferDesc desc_;
    uint32_t priority_ = 0;
    id<MTLBuffer> baseBuffer_ = nil;
    id<MTLBuffer> activeBuffer_ = nil;
    size_t activeOffset_ = 0;
    uint64_t lastUsedSeq_ = 0;
    std::shared_ptr<MtlSharedBufferPool> bufPool_;
    std::vector<MtlBufferDynSlot> dynSlots_;
    int32_t activeDynSlot_ = -1;
};

class MtlIndexBuffer9 final : public MtlComObject<MtlIndexBuffer9, IMtlDirect3DIndexBuffer9> {
public:
    MtlIndexBuffer9(MtlDirect3DDevice9 *dev, uint32_t length, uint32_t usage, uint32_t format, uint32_t pool);
    ~MtlIndexBuffer9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT SetPrivateData(const D3DMtlGuid *, const void *, uint32_t, uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetPrivateData(const D3DMtlGuid *, void *, uint32_t *) override { return D3D_ERR_NOTAVAILABLE; }
    MTL_HRESULT FreePrivateData(const D3DMtlGuid *) override { return D3D_S_OK; }
    uint32_t    SetPriority(uint32_t p) override { uint32_t old = priority_; priority_ = p; return old; }
    uint32_t    GetPriority() override { return priority_; }
    void        PreLoad() override {}
    uint32_t    GetType() override { return D3DM_RTYPE_INDEXBUFFER; }
    MTL_HRESULT Lock(uint32_t offset, uint32_t size, void **ppbData, uint32_t flags) override;
    MTL_HRESULT Unlock() override;
    MTL_HRESULT GetDesc(D3DMtlBufferDesc *pDesc) override;

    id<MTLBuffer> activeBuffer() const { return activeBuffer_; }
    size_t        activeOffset() const { return activeOffset_; }
    MTLIndexType  indexType() const { return desc_.Format == D3DM_FMT_INDEX32 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16; }
    uint32_t      indexStride() const { return desc_.Format == D3DM_FMT_INDEX32 ? 4u : 2u; }
    void          markUsed(uint64_t cmdSeq);

private:
    MtlDirect3DDevice9 *device_;
    D3DMtlBufferDesc desc_;
    uint32_t priority_ = 0;
    id<MTLBuffer> baseBuffer_ = nil;
    id<MTLBuffer> activeBuffer_ = nil;
    size_t activeOffset_ = 0;
    uint64_t lastUsedSeq_ = 0;
    std::shared_ptr<MtlSharedBufferPool> bufPool_;
    std::vector<MtlBufferDynSlot> dynSlots_;
    int32_t activeDynSlot_ = -1;
};

/* ---- Surfaces & Textures ---- */
class MtlSurface9 final : public MtlComObject<MtlSurface9, IMtlDirect3DSurface9> {
public:
    /* Standalone GPU / SysMem / NULL surface */
    MtlSurface9(MtlDirect3DDevice9 *dev, uint32_t w, uint32_t h, uint32_t format,
                uint32_t usage, uint32_t pool, uint32_t msType, uint32_t msQual,
                id<MTLTexture> existingTex = nil, IMtlUnknown *container = nullptr,
                uint32_t mipLevel = 0, uint32_t slice = 0);
    ~MtlSurface9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    uint32_t    AddRef() override;
    uint32_t    Release() override;
    MTL_HRESULT SetPrivateData(const D3DMtlGuid *, const void *, uint32_t, uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetPrivateData(const D3DMtlGuid *, void *, uint32_t *) override { return D3D_ERR_NOTAVAILABLE; }
    MTL_HRESULT FreePrivateData(const D3DMtlGuid *) override { return D3D_S_OK; }
    uint32_t    SetPriority(uint32_t p) override { uint32_t old = priority_; priority_ = p; return old; }
    uint32_t    GetPriority() override { return priority_; }
    void        PreLoad() override {}
    uint32_t    GetType() override { return D3DM_RTYPE_SURFACE; }
    MTL_HRESULT GetContainer(const D3DMtlGuid *riid, void **ppContainer) override;
    MTL_HRESULT GetDesc(D3DMtlSurfaceDesc *pDesc) override;
    MTL_HRESULT LockRect(D3DMtlLockedRect *pLockedRect, const D3DMtlRect *pRect, uint32_t Flags) override;
    MTL_HRESULT UnlockRect() override;
    MTL_HRESULT GetDC(void **) override { return D3D_ERR_INVALIDCALL; }
    MTL_HRESULT ReleaseDC(void *) override { return D3D_ERR_INVALIDCALL; }

    id<MTLTexture> texture() const { return texture_; }
    id<MTLTexture> srgbRenderTargetView();
    const MtlFormatInfo &formatInfo() const { return fmtInfo_; }
    const D3DMtlSurfaceDesc &desc() const { return desc_; }
    uint32_t mipLevel() const { return mipLevel_; }
    uint32_t slice() const { return slice_; }
    bool isNull() const { return fmtInfo_.isNullFormat; }
    bool isSysMem() const { return desc_.Pool == D3DM_POOL_SYSTEMMEM || desc_.Pool == D3DM_POOL_SCRATCH; }
    std::vector<uint8_t> &cpuData() { return cpuData_; }
    void ensureCpuBacking();
    void uploadCpuToGpu(const D3DMtlRect *rect = nullptr);
    void replaceTexture(id<MTLTexture> newTex, uint32_t w, uint32_t h, uint32_t format);

    bool hasPendingColorClear() const { return pendingColorClear_; }
    bool hasPendingZClear() const { return pendingZClear_; }
    bool hasPendingSClear() const { return pendingSClear_; }
    bool hasAnyPendingClear() const { return pendingColorClear_ || pendingZClear_ || pendingSClear_; }
    D3DCOLOR_T pendingClearColor() const { return pendingClearColor_; }
    float pendingClearZ() const { return pendingClearZ_; }
    uint32_t pendingClearStencil() const { return pendingClearStencil_; }
    void setPendingColorClear(D3DCOLOR_T c) { pendingColorClear_ = true; pendingClearColor_ = c; }
    void setPendingZClear(float z) { pendingZClear_ = true; pendingClearZ_ = z; }
    void setPendingSClear(uint32_t s) { pendingSClear_ = true; pendingClearStencil_ = s; }
    void clearPendingColorClear() { pendingColorClear_ = false; }
    void clearPendingZClear() { pendingZClear_ = false; }
    void clearPendingSClear() { pendingSClear_ = false; }
    void clearAllPendingClears() { pendingColorClear_ = pendingZClear_ = pendingSClear_ = false; }
    bool inPendingClearList() const { return inPendingClearList_; }
    void setInPendingClearList(bool v) { inPendingClearList_ = v; }
    bool hasGpuData() const { return hasGpuData_; }
    void setHasGpuData(bool v) { hasGpuData_ = v; }
    bool cpuDataBacked() const { return cpuDataBacked_; }
    void setCpuDataBacked(bool v) { cpuDataBacked_ = v; }

private:
    MtlDirect3DDevice9 *device_;
    IMtlUnknown *container_ = nullptr;
    D3DMtlSurfaceDesc desc_;
    MtlFormatInfo fmtInfo_;
    uint32_t mipLevel_ = 0;
    uint32_t slice_ = 0;
    uint32_t priority_ = 0;
    id<MTLTexture> texture_ = nil;
    id<MTLTexture> srgbView_ = nil;
    std::vector<uint8_t> cpuData_;
    MtlRingBuffer::Allocation stagedLockAlloc_;
    uint32_t stagedRowPitch_ = 0;
    uint32_t stagedBytes_ = 0;
    D3DMtlRect lockedRect_ = {};
    bool hasLockedRect_ = false;
    bool lockReadOnly_ = false;
    bool hasGpuData_ = false;
    bool cpuDataBacked_ = false;
    bool pendingColorClear_ = false;
    bool pendingZClear_ = false;
    bool pendingSClear_ = false;
    bool inPendingClearList_ = false;
    D3DCOLOR_T pendingClearColor_ = 0;
    float pendingClearZ_ = 1.0f;
    uint32_t pendingClearStencil_ = 0;
};

/* Common base for 2D / Cube / Volume textures */
struct MtlTextureCommon {
    MtlDirect3DDevice9 *device = nullptr;
    id<MTLTexture> baseTexture = nil;
    id<MTLTexture> sampleViewLinear = nil;
    id<MTLTexture> sampleViewSrgb = nil;
    MtlFormatInfo  fmtInfo = {};
    uint32_t d3dFormat = 0;
    uint32_t usage = 0;
    uint32_t pool = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t levels = 1;
    uint32_t slices = 1;
    uint32_t lod = 0;
    uint32_t priority = 0;
    uint32_t autoGenFilter = D3DM_TEXF_LINEAR;
    MTLTextureType mtlType = MTLTextureType2D;

    id<MTLTexture> getSampleView(bool srgb);
};

class MtlTexture9 final : public MtlComObject<MtlTexture9, IMtlDirect3DTexture9> {
public:
    MtlTexture9(MtlDirect3DDevice9 *dev, uint32_t w, uint32_t h, uint32_t levels,
                uint32_t usage, uint32_t format, uint32_t pool);
    ~MtlTexture9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT SetPrivateData(const D3DMtlGuid *, const void *, uint32_t, uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetPrivateData(const D3DMtlGuid *, void *, uint32_t *) override { return D3D_ERR_NOTAVAILABLE; }
    MTL_HRESULT FreePrivateData(const D3DMtlGuid *) override { return D3D_S_OK; }
    uint32_t    SetPriority(uint32_t p) override { uint32_t old = tex_.priority; tex_.priority = p; return old; }
    uint32_t    GetPriority() override { return tex_.priority; }
    void        PreLoad() override {}
    uint32_t    GetType() override { return D3DM_RTYPE_TEXTURE; }
    uint32_t    SetLOD(uint32_t l) override { uint32_t old = tex_.lod; tex_.lod = (tex_.levels > 0) ? std::min(l, tex_.levels - 1u) : 0u; return old; }
    uint32_t    GetLOD() override { return tex_.lod; }
    uint32_t    GetLevelCount() override { return tex_.levels; }
    MTL_HRESULT SetAutoGenFilterType(uint32_t f) override { tex_.autoGenFilter = f; return D3D_S_OK; }
    uint32_t    GetAutoGenFilterType() override { return tex_.autoGenFilter; }
    void        GenerateMipSubLevels() override;
    MTL_HRESULT GetLevelDesc(uint32_t Level, D3DMtlSurfaceDesc *pDesc) override;
    MTL_HRESULT GetSurfaceLevel(uint32_t Level, IMtlDirect3DSurface9 **ppSurfaceLevel) override;
    MTL_HRESULT LockRect(uint32_t Level, D3DMtlLockedRect *pLockedRect, const D3DMtlRect *pRect, uint32_t Flags) override;
    MTL_HRESULT UnlockRect(uint32_t Level) override;
    MTL_HRESULT AddDirtyRect(const D3DMtlRect *) override { return D3D_S_OK; }

    MtlTextureCommon &common() { return tex_; }
    MtlSurface9 *surfaceAt(uint32_t level);

private:
    MtlTextureCommon tex_;
    std::vector<MtlSurface9 *> surfaces_;
};

class MtlCubeTexture9 final : public MtlComObject<MtlCubeTexture9, IMtlDirect3DCubeTexture9> {
public:
    MtlCubeTexture9(MtlDirect3DDevice9 *dev, uint32_t edge, uint32_t levels,
                    uint32_t usage, uint32_t format, uint32_t pool);
    ~MtlCubeTexture9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT SetPrivateData(const D3DMtlGuid *, const void *, uint32_t, uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetPrivateData(const D3DMtlGuid *, void *, uint32_t *) override { return D3D_ERR_NOTAVAILABLE; }
    MTL_HRESULT FreePrivateData(const D3DMtlGuid *) override { return D3D_S_OK; }
    uint32_t    SetPriority(uint32_t p) override { uint32_t old = tex_.priority; tex_.priority = p; return old; }
    uint32_t    GetPriority() override { return tex_.priority; }
    void        PreLoad() override {}
    uint32_t    GetType() override { return D3DM_RTYPE_CUBETEXTURE; }
    uint32_t    SetLOD(uint32_t l) override { uint32_t old = tex_.lod; tex_.lod = (tex_.levels > 0) ? std::min(l, tex_.levels - 1u) : 0u; return old; }
    uint32_t    GetLOD() override { return tex_.lod; }
    uint32_t    GetLevelCount() override { return tex_.levels; }
    MTL_HRESULT SetAutoGenFilterType(uint32_t f) override { tex_.autoGenFilter = f; return D3D_S_OK; }
    uint32_t    GetAutoGenFilterType() override { return tex_.autoGenFilter; }
    void        GenerateMipSubLevels() override;
    MTL_HRESULT GetLevelDesc(uint32_t Level, D3DMtlSurfaceDesc *pDesc) override;
    MTL_HRESULT GetCubeMapSurface(uint32_t FaceType, uint32_t Level, IMtlDirect3DSurface9 **ppCubeMapSurface) override;
    MTL_HRESULT LockRect(uint32_t FaceType, uint32_t Level, D3DMtlLockedRect *pLockedRect, const D3DMtlRect *pRect, uint32_t Flags) override;
    MTL_HRESULT UnlockRect(uint32_t FaceType, uint32_t Level) override;
    MTL_HRESULT AddDirtyRect(uint32_t, const D3DMtlRect *) override { return D3D_S_OK; }

    MtlTextureCommon &common() { return tex_; }
    MtlSurface9 *surfaceAt(uint32_t face, uint32_t level);

private:
    MtlTextureCommon tex_;
    std::vector<MtlSurface9 *> surfaces_; /* [face * levels + level] */
};

class MtlVolume9 final : public MtlComObject<MtlVolume9, IMtlDirect3DVolume9> {
public:
    MtlVolume9(MtlDirect3DDevice9 *dev, IMtlUnknown *container, id<MTLTexture> tex,
               uint32_t w, uint32_t h, uint32_t d, uint32_t format, uint32_t usage, uint32_t pool, uint32_t level);
    ~MtlVolume9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    uint32_t    AddRef() override;
    uint32_t    Release() override;
    MTL_HRESULT SetPrivateData(const D3DMtlGuid *, const void *, uint32_t, uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetPrivateData(const D3DMtlGuid *, void *, uint32_t *) override { return D3D_ERR_NOTAVAILABLE; }
    MTL_HRESULT FreePrivateData(const D3DMtlGuid *) override { return D3D_S_OK; }
    MTL_HRESULT GetContainer(const D3DMtlGuid *riid, void **ppContainer) override;
    MTL_HRESULT GetDesc(D3DMtlVolumeDesc *pDesc) override;
    MTL_HRESULT LockBox(D3DMtlLockedBox *pLockedVolume, const D3DMtlBox *pBox, uint32_t Flags) override;
    MTL_HRESULT UnlockBox() override;

    std::vector<uint8_t> &cpuData() { return cpuData_; }
    const D3DMtlVolumeDesc &desc() const { return desc_; }
    void ensureCpuBacking();
    void uploadCpuToGpu();

private:
    MtlDirect3DDevice9 *device_;
    IMtlUnknown *container_;
    id<MTLTexture> texture_;
    D3DMtlVolumeDesc desc_;
    uint32_t mipLevel_;
    std::vector<uint8_t> cpuData_;
    bool lockReadOnly_ = false;
};

class MtlVolumeTexture9 final : public MtlComObject<MtlVolumeTexture9, IMtlDirect3DVolumeTexture9> {
public:
    MtlVolumeTexture9(MtlDirect3DDevice9 *dev, uint32_t w, uint32_t h, uint32_t d,
                      uint32_t levels, uint32_t usage, uint32_t format, uint32_t pool);
    ~MtlVolumeTexture9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT SetPrivateData(const D3DMtlGuid *, const void *, uint32_t, uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetPrivateData(const D3DMtlGuid *, void *, uint32_t *) override { return D3D_ERR_NOTAVAILABLE; }
    MTL_HRESULT FreePrivateData(const D3DMtlGuid *) override { return D3D_S_OK; }
    uint32_t    SetPriority(uint32_t p) override { uint32_t old = tex_.priority; tex_.priority = p; return old; }
    uint32_t    GetPriority() override { return tex_.priority; }
    void        PreLoad() override {}
    uint32_t    GetType() override { return D3DM_RTYPE_VOLUMETEXTURE; }
    uint32_t    SetLOD(uint32_t l) override { uint32_t old = tex_.lod; tex_.lod = (tex_.levels > 0) ? std::min(l, tex_.levels - 1u) : 0u; return old; }
    uint32_t    GetLOD() override { return tex_.lod; }
    uint32_t    GetLevelCount() override { return tex_.levels; }
    MTL_HRESULT SetAutoGenFilterType(uint32_t f) override { tex_.autoGenFilter = f; return D3D_S_OK; }
    uint32_t    GetAutoGenFilterType() override { return tex_.autoGenFilter; }
    void        GenerateMipSubLevels() override {}
    MTL_HRESULT GetLevelDesc(uint32_t Level, D3DMtlVolumeDesc *pDesc) override;
    MTL_HRESULT GetVolumeLevel(uint32_t Level, IMtlDirect3DVolume9 **ppVolumeLevel) override;
    MTL_HRESULT LockBox(uint32_t Level, D3DMtlLockedBox *pLockedVolume, const D3DMtlBox *pBox, uint32_t Flags) override;
    MTL_HRESULT UnlockBox(uint32_t Level) override;
    MTL_HRESULT AddDirtyBox(const D3DMtlBox *) override { return D3D_S_OK; }

    MtlTextureCommon &common() { return tex_; }
    MtlVolume9 *volumeAt(uint32_t level);

private:
    MtlTextureCommon tex_;
    std::vector<MtlVolume9 *> volumes_;
};

/* Helper: retrieve MtlTextureCommon* from any IMtlDirect3DBaseTexture9* */
MtlTextureCommon *getTextureCommon(IMtlDirect3DBaseTexture9 *baseTex);

/* ---- Vertex Declaration, Shaders, Queries ---- */
class MtlVertexDeclaration9 final : public MtlComObject<MtlVertexDeclaration9, IMtlDirect3DVertexDeclaration9> {
public:
    MtlVertexDeclaration9(MtlDirect3DDevice9 *dev, const D3DMtlVertexElement *elems);
    ~MtlVertexDeclaration9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT GetDeclaration(D3DMtlVertexElement *pElement, uint32_t *pNumElements) override;

    const std::vector<D3DMtlVertexElement> &elements() const { return elements_; }
    uint64_t hash() const { return hash_; }

private:
    MtlDirect3DDevice9 *device_;
    std::vector<D3DMtlVertexElement> elements_; /* excludes D3DDECL_END */
    uint64_t hash_ = 0;
};

class MtlVertexShader9 final : public MtlComObject<MtlVertexShader9, IMtlDirect3DVertexShader9> {
public:
    MtlVertexShader9(MtlDirect3DDevice9 *dev, std::shared_ptr<MtlParsedShader> shader);
    ~MtlVertexShader9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT GetFunction(void *pData, uint32_t *pSizeOfData) override;

    MtlParsedShader *parsed() const { return shader_.get(); }

private:
    MtlDirect3DDevice9 *device_;
    std::shared_ptr<MtlParsedShader> shader_;
};

class MtlPixelShader9 final : public MtlComObject<MtlPixelShader9, IMtlDirect3DPixelShader9> {
public:
    MtlPixelShader9(MtlDirect3DDevice9 *dev, MtlShaderCompiler *compiler,
                    std::shared_ptr<MtlAsyncParsedShader> handle);
    ~MtlPixelShader9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT GetFunction(void *pData, uint32_t *pSizeOfData) override;

    MtlParsedShader *parsed() const;
    bool hasReadyParsed() const {
        return asyncHandle_ && asyncHandle_->readyPtr.load(std::memory_order_acquire) != nullptr;
    }

private:
    MtlDirect3DDevice9 *device_;
    MtlShaderCompiler *compiler_;
    std::shared_ptr<MtlAsyncParsedShader> asyncHandle_;
};

class MtlQuery9 final : public MtlComObject<MtlQuery9, IMtlDirect3DQuery9> {
public:
    MtlQuery9(MtlDirect3DDevice9 *dev, uint32_t type);
    ~MtlQuery9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    uint32_t    GetType() override { return type_; }
    uint32_t    GetDataSize() override { return 4u; }
    MTL_HRESULT Issue(uint32_t dwIssueFlags) override;
    MTL_HRESULT GetData(void *pData, uint32_t dwSize, uint32_t dwGetDataFlags) override;

    uint32_t visibilitySlot() const { return visSlot_; }

private:
    MtlDirect3DDevice9 *device_;
    uint32_t type_;
    uint32_t visSlot_ = 0xFFFFFFFFu;
    uint64_t issuedSeq_ = 0;
    uint32_t consecutiveMisses_ = 0;
    bool active_ = false;
    bool issued_ = false;
};

} // namespace mtl_d3d9
