#include "mtl_d3d9_device.h"

#import <AppKit/AppKit.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__attribute__((weak)) extern "C" int host_cursor_pos(int *x, int *y);

namespace mtl_d3d9 {

static std::mutex gWarmDevMu;
static MtlDirect3DDevice9 *gWarmDev = nullptr;
static std::string gPendingWarmMap;

static uint64_t hashMapStem(const char *name) {
    if (!name || !*name) return 0;
    uint64_t h = 14695981039346656037ull;
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(name); *p; ++p) {
        uint8_t c = uint8_t(std::tolower(*p));
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

static void registerWarmDevice(MtlDirect3DDevice9 *dev) {
    std::string pending;
    {
        std::lock_guard<std::mutex> lk(gWarmDevMu);
        gWarmDev = dev;
        pending = gPendingWarmMap;
    }
    if (dev && !pending.empty()) {
        dev->onWarmSetMap(pending.c_str());
    }
}

static void unregisterWarmDevice(MtlDirect3DDevice9 *dev) {
    std::lock_guard<std::mutex> lk(gWarmDevMu);
    if (gWarmDev == dev) gWarmDev = nullptr;
}

static inline float u32ToFloat(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

static inline uint32_t floatToU32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

static inline double srgbByteToLinear(uint8_t v) {
    double c = double(v) / 255.0;
    return (c <= 0.04045) ? (c / 12.92) : std::pow((c + 0.055) / 1.055, 2.4);
}

static uint32_t primCountToVertexCount(uint32_t primType, uint32_t primCount) {
    switch (primType) {
        case D3DM_PT_POINTLIST:     return primCount;
        case D3DM_PT_LINELIST:      return primCount * 2u;
        case D3DM_PT_LINESTRIP:     return primCount + 1u;
        case D3DM_PT_TRIANGLELIST:  return primCount * 3u;
        case D3DM_PT_TRIANGLESTRIP: return primCount + 2u;
        case D3DM_PT_TRIANGLEFAN:   return primCount + 2u;
        default:                    return 0u;
    }
}

static MTLPrimitiveType mapD3dPrimType(uint32_t primType) {
    switch (primType) {
        case D3DM_PT_POINTLIST:     return MTLPrimitiveTypePoint;
        case D3DM_PT_LINELIST:      return MTLPrimitiveTypeLine;
        case D3DM_PT_LINESTRIP:     return MTLPrimitiveTypeLineStrip;
        case D3DM_PT_TRIANGLELIST:  return MTLPrimitiveTypeTriangle;
        case D3DM_PT_TRIANGLESTRIP: return MTLPrimitiveTypeTriangleStrip;
        case D3DM_PT_TRIANGLEFAN:   return MTLPrimitiveTypeTriangle; /* converted to tri list */
        default:                    return MTLPrimitiveTypeTriangle;
    }
}

static uint64_t hashStreamLayout(const MtlStreamBindingInfo streams[16]) {
    const uint8_t *p = reinterpret_cast<const uint8_t *>(streams);
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < sizeof(MtlStreamBindingInfo) * 16; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

/* ========================================================================= */
/* MtlDeviceState                                                            */
/* ========================================================================= */

void MtlDeviceState::resetDefaults(uint32_t bbW, uint32_t bbH) {
    releaseRefs();

    std::memset(renderStates, 0, sizeof(renderStates));
    renderStates[D3DM_RS_ZENABLE]                  = 1;
    renderStates[D3DM_RS_FILLMODE]                 = D3DM_FILL_SOLID;
    renderStates[D3DM_RS_SHADEMODE]                = 2; /* GOURAUD */
    renderStates[D3DM_RS_ZWRITEENABLE]             = 1;
    renderStates[D3DM_RS_ALPHATESTENABLE]          = 0;
    renderStates[D3DM_RS_SRCBLEND]                 = D3DM_BLEND_ONE;
    renderStates[D3DM_RS_DESTBLEND]                = D3DM_BLEND_ZERO;
    renderStates[D3DM_RS_CULLMODE]                 = D3DM_CULL_CCW;
    renderStates[D3DM_RS_ZFUNC]                    = D3DM_CMP_LESSEQUAL;
    renderStates[D3DM_RS_ALPHAREF]                 = 0;
    renderStates[D3DM_RS_ALPHAFUNC]                = D3DM_CMP_ALWAYS;
    renderStates[D3DM_RS_ALPHABLENDENABLE]         = 0;
    renderStates[D3DM_RS_FOGENABLE]                = 0;
    renderStates[D3DM_RS_FOGCOLOR]                 = 0;
    renderStates[D3DM_RS_FOGTABLEMODE]             = 0;
    renderStates[D3DM_RS_FOGSTART]                 = floatToU32(0.0f);
    renderStates[D3DM_RS_FOGEND]                   = floatToU32(1.0f);
    renderStates[D3DM_RS_FOGDENSITY]               = floatToU32(1.0f);
    renderStates[D3DM_RS_STENCILENABLE]            = 0;
    renderStates[D3DM_RS_STENCILFAIL]              = D3DM_STENCILOP_KEEP;
    renderStates[D3DM_RS_STENCILZFAIL]             = D3DM_STENCILOP_KEEP;
    renderStates[D3DM_RS_STENCILPASS]              = D3DM_STENCILOP_KEEP;
    renderStates[D3DM_RS_STENCILFUNC]              = D3DM_CMP_ALWAYS;
    renderStates[D3DM_RS_STENCILREF]               = 0;
    renderStates[D3DM_RS_STENCILMASK]              = 0xFFFFFFFFu;
    renderStates[D3DM_RS_STENCILWRITEMASK]         = 0xFFFFFFFFu;
    renderStates[D3DM_RS_CLIPPING]                 = 1;
    renderStates[D3DM_RS_CLIPPLANEENABLE]          = 0;
    renderStates[D3DM_RS_POINTSIZE]                = floatToU32(1.0f);
    renderStates[D3DM_RS_POINTSIZE_MIN]            = floatToU32(1.0f);
    renderStates[D3DM_RS_POINTSPRITEENABLE]        = 0;
    renderStates[D3DM_RS_POINTSIZE_MAX]            = floatToU32(64.0f);
    renderStates[D3DM_RS_COLORWRITEENABLE]         = 0xFu;
    renderStates[D3DM_RS_COLORWRITEENABLE1]        = 0xFu;
    renderStates[D3DM_RS_COLORWRITEENABLE2]        = 0xFu;
    renderStates[D3DM_RS_COLORWRITEENABLE3]        = 0xFu;
    renderStates[D3DM_RS_BLENDOP]                  = D3DM_BLENDOP_ADD;
    renderStates[D3DM_RS_SCISSORTESTENABLE]        = 0;
    renderStates[D3DM_RS_SLOPESCALEDEPTHBIAS]      = 0;
    renderStates[D3DM_RS_TWOSIDEDSTENCILMODE]      = 0;
    renderStates[D3DM_RS_CCW_STENCILFAIL]          = D3DM_STENCILOP_KEEP;
    renderStates[D3DM_RS_CCW_STENCILZFAIL]         = D3DM_STENCILOP_KEEP;
    renderStates[D3DM_RS_CCW_STENCILPASS]          = D3DM_STENCILOP_KEEP;
    renderStates[D3DM_RS_CCW_STENCILFUNC]          = D3DM_CMP_ALWAYS;
    renderStates[D3DM_RS_BLENDFACTOR]              = 0xFFFFFFFFu;
    renderStates[D3DM_RS_SRGBWRITEENABLE]          = 0;
    renderStates[D3DM_RS_DEPTHBIAS]                = 0;
    renderStates[D3DM_RS_SEPARATEALPHABLENDENABLE] = 0;
    renderStates[D3DM_RS_SRCBLENDALPHA]            = D3DM_BLEND_ONE;
    renderStates[D3DM_RS_DESTBLENDALPHA]           = D3DM_BLEND_ZERO;
    renderStates[D3DM_RS_BLENDOPALPHA]             = D3DM_BLENDOP_ADD;

    for (uint32_t s = 0; s < kMaxSamplers; s++) {
        std::memset(samplerStates[s], 0, sizeof(samplerStates[s]));
        samplerStates[s][D3DM_SAMP_ADDRESSU]      = D3DM_TADDRESS_WRAP;
        samplerStates[s][D3DM_SAMP_ADDRESSV]      = D3DM_TADDRESS_WRAP;
        samplerStates[s][D3DM_SAMP_ADDRESSW]      = D3DM_TADDRESS_WRAP;
        samplerStates[s][D3DM_SAMP_BORDERCOLOR]   = 0;
        samplerStates[s][D3DM_SAMP_MAGFILTER]     = D3DM_TEXF_POINT;
        samplerStates[s][D3DM_SAMP_MINFILTER]     = D3DM_TEXF_POINT;
        samplerStates[s][D3DM_SAMP_MIPFILTER]     = D3DM_TEXF_NONE;
        samplerStates[s][D3DM_SAMP_MIPMAPLODBIAS] = 0;
        samplerStates[s][D3DM_SAMP_MAXMIPLEVEL]   = 0;
        samplerStates[s][D3DM_SAMP_MAXANISOTROPY] = 1;
        samplerStates[s][D3DM_SAMP_SRGBTEXTURE]   = 0;
    }

    std::memset(texStageStates, 0, sizeof(texStageStates));
    fetch4Mask = 0;
    fvf = 0;
    for (uint32_t i = 0; i < 16; i++) {
        streams[i] = {};
        streams[i].freq = 1;
    }

    viewport = { 0, 0, bbW, bbH, 0.0f, 1.0f };
    scissorRect = { 0, 0, int32_t(bbW), int32_t(bbH) };
    std::memset(clipPlanes, 0, sizeof(clipPlanes));
    std::memset(vsConstF, 0, sizeof(vsConstF));
    std::memset(vsConstI, 0, sizeof(vsConstI));
    std::memset(vsConstB, 0, sizeof(vsConstB));
    std::memset(psConstF, 0, sizeof(psConstF));
    std::memset(psConstI, 0, sizeof(psConstI));
    std::memset(psConstB, 0, sizeof(psConstB));
    std::memset(transforms, 0, sizeof(transforms));
    for (int i = 0; i < 4; i++) {
        transforms[i][0] = transforms[i][5] = transforms[i][10] = transforms[i][15] = 1.0f;
    }
}

void MtlDeviceState::releaseRefs() {
    for (uint32_t s = 0; s < kMaxSamplers; s++) {
        if (textures[s]) { textures[s]->Release(); textures[s] = nullptr; }
    }
    if (vs) { vs->Release(); vs = nullptr; }
    if (ps) { ps->Release(); ps = nullptr; }
    if (vdecl) { vdecl->Release(); vdecl = nullptr; }
    for (uint32_t i = 0; i < 16; i++) {
        if (streams[i].vb) { streams[i].vb->Release(); streams[i].vb = nullptr; }
    }
    if (ib) { ib->Release(); ib = nullptr; }
}

/* ========================================================================= */
/* MtlStateBlock9                                                            */
/* ========================================================================= */

MtlStateBlock9::MtlStateBlock9(MtlDirect3DDevice9 *dev, uint32_t type)
    : device_(dev), type_(type) {
    if (type == 1 /* D3DSBT_ALL */) {
        captureAllPixel_ = true;
        captureAllVertex_ = true;
        Capture();
    } else if (type == 2 /* D3DSBT_PIXELSTATE */) {
        captureAllPixel_ = true;
        Capture();
    } else if (type == 3 /* D3DSBT_VERTEXSTATE */) {
        captureAllVertex_ = true;
        Capture();
    }
}

MtlStateBlock9::~MtlStateBlock9() {
    clearRecorded();
}

void MtlStateBlock9::clearRecorded() {
    rs_.clear();
    ss_.clear();
    tss_.clear();
    for (uint32_t i = 0; i < kMaxSamplers; i++) {
        if (textures_[i]) { textures_[i]->Release(); textures_[i] = nullptr; }
    }
    texMask_ = 0;
    if (vs_) { vs_->Release(); vs_ = nullptr; }
    if (ps_) { ps_->Release(); ps_ = nullptr; }
    if (vdecl_) { vdecl_->Release(); vdecl_ = nullptr; }
    for (uint32_t i = 0; i < 16; i++) {
        if (streams_[i].vb) { streams_[i].vb->Release(); streams_[i].vb = nullptr; }
    }
    if (ib_) { ib_->Release(); ib_ = nullptr; }
    hasVs_ = hasPs_ = hasDecl_ = hasFvf_ = hasIb_ = hasVp_ = hasScissor_ = false;
    streamMask_ = streamFreqMask_ = clipMask_ = 0;
    std::memset(vsConstFMask_, 0, sizeof(vsConstFMask_));
    std::memset(psConstFMask_, 0, sizeof(psConstFMask_));
    vsConstIMask_ = vsConstBMask_ = psConstIMask_ = psConstBMask_ = 0;
}

MTL_HRESULT MtlStateBlock9::GetDevice(IMtlDirect3DDevice9 **ppDevice) {
    if (!ppDevice) return D3D_ERR_INVALIDCALL;
    device_->AddRef();
    *ppDevice = device_;
    return D3D_S_OK;
}

MTL_HRESULT MtlStateBlock9::Capture() {
    const MtlDeviceState &st = device_->state();
    if (captureAllPixel_ || captureAllVertex_) {
        clearRecorded();
        for (uint32_t r = 0; r < D3DM_RS_MAX; r++) rs_.push_back({r, st.renderStates[r]});
        if (captureAllPixel_) {
            for (uint32_t s = 0; s < 16; s++) {
                for (uint32_t t = 1; t < D3DM_SAMP_MAX; t++) ss_.push_back({(s << 8) | t, st.samplerStates[s][t]});
                recordTexture(s, st.textures[s]);
            }
            for (uint32_t s = 0; s < kMaxTexStages; s++) {
                for (uint32_t t = 1; t < kMaxTexStageStates; t++) tss_.push_back({(s << 8) | t, st.texStageStates[s][t]});
            }
            recordPs(st.ps);
            recordPsConstF(0, &st.psConstF[0][0], 224);
            recordPsConstI(0, &st.psConstI[0][0], 16);
            recordPsConstB(0, st.psConstB, 16);
        }
        if (captureAllVertex_) {
            for (uint32_t s = 16; s < kMaxSamplers; s++) {
                for (uint32_t t = 1; t < D3DM_SAMP_MAX; t++) ss_.push_back({(s << 8) | t, st.samplerStates[s][t]});
                recordTexture(s, st.textures[s]);
            }
            recordVs(st.vs);
            recordVdecl(st.vdecl);
            recordFvf(st.fvf);
            for (uint32_t i = 0; i < 16; i++) {
                recordStream(i, st.streams[i].vb, st.streams[i].offset, st.streams[i].stride);
                recordStreamFreq(i, st.streams[i].freq);
            }
            recordIndices(st.ib);
            recordViewport(st.viewport);
            recordScissor(st.scissorRect);
            for (uint32_t i = 0; i < 6; i++) recordClipPlane(i, st.clipPlanes[i]);
            recordVsConstF(0, &st.vsConstF[0][0], 256);
            recordVsConstI(0, &st.vsConstI[0][0], 16);
            recordVsConstB(0, st.vsConstB, 16);
        }
        return D3D_S_OK;
    }

    /* Update only previously recorded fields */
    for (auto &kv : rs_) kv.second = st.renderStates[kv.first];
    for (auto &kv : ss_) kv.second = st.samplerStates[kv.first >> 8][kv.first & 0xFF];
    for (auto &kv : tss_) kv.second = st.texStageStates[kv.first >> 8][kv.first & 0xFF];
    for (uint32_t s = 0; s < kMaxSamplers; s++) {
        if (texMask_ & (1u << s)) recordTexture(s, st.textures[s]);
    }
    if (hasVs_) recordVs(st.vs);
    if (hasPs_) recordPs(st.ps);
    if (hasDecl_) recordVdecl(st.vdecl);
    if (hasFvf_) fvf_ = st.fvf;
    for (uint32_t i = 0; i < 16; i++) {
        if (streamMask_ & (1u << i)) recordStream(i, st.streams[i].vb, st.streams[i].offset, st.streams[i].stride);
        if (streamFreqMask_ & (1u << i)) streams_[i].freq = st.streams[i].freq;
    }
    if (hasIb_) recordIndices(st.ib);
    if (hasVp_) vp_ = st.viewport;
    if (hasScissor_) scissor_ = st.scissorRect;
    for (uint32_t i = 0; i < 6; i++) {
        if (clipMask_ & (1u << i)) std::memcpy(clipPlanes_[i], st.clipPlanes[i], 16);
    }
    for (uint32_t i = 0; i < 256; i++) if (vsConstFMask_[i]) std::memcpy(vsConstF_[i], st.vsConstF[i], 16);
    for (uint32_t i = 0; i < 16; i++) {
        if (vsConstIMask_ & (1u << i)) std::memcpy(vsConstI_[i], st.vsConstI[i], 16);
        if (vsConstBMask_ & (1u << i)) vsConstB_[i] = st.vsConstB[i];
        if (psConstIMask_ & (1u << i)) std::memcpy(psConstI_[i], st.psConstI[i], 16);
        if (psConstBMask_ & (1u << i)) psConstB_[i] = st.psConstB[i];
    }
    for (uint32_t i = 0; i < 224; i++) if (psConstFMask_[i]) std::memcpy(psConstF_[i], st.psConstF[i], 16);
    return D3D_S_OK;
}

MTL_HRESULT MtlStateBlock9::Apply() {
    for (const auto &kv : rs_) device_->SetRenderState(kv.first, kv.second);
    for (const auto &kv : ss_) {
        uint32_t slot = kv.first >> 8;
        uint32_t d3dSamp = (slot < 16) ? slot : (D3DM_VERTEXTEXTURESAMPLER0 + (slot - 16));
        device_->SetSamplerState(d3dSamp, kv.first & 0xFF, kv.second);
    }
    for (const auto &kv : tss_) device_->SetTextureStageState(kv.first >> 8, kv.first & 0xFF, kv.second);
    for (uint32_t s = 0; s < kMaxSamplers; s++) {
        if (texMask_ & (1u << s)) {
            uint32_t d3dSamp = (s < 16) ? s : (D3DM_VERTEXTEXTURESAMPLER0 + (s - 16));
            device_->SetTexture(d3dSamp, textures_[s]);
        }
    }
    if (hasVs_) device_->SetVertexShader(vs_);
    if (hasPs_) device_->SetPixelShader(ps_);
    if (hasDecl_) device_->SetVertexDeclaration(vdecl_);
    if (hasFvf_) device_->SetFVF(fvf_);
    for (uint32_t i = 0; i < 16; i++) {
        if (streamMask_ & (1u << i)) device_->SetStreamSource(i, streams_[i].vb, streams_[i].offset, streams_[i].stride);
        if (streamFreqMask_ & (1u << i)) device_->SetStreamSourceFreq(i, streams_[i].freq);
    }
    if (hasIb_) device_->SetIndices(ib_);
    if (hasVp_) device_->SetViewport(&vp_);
    if (hasScissor_) device_->SetScissorRect(&scissor_);
    for (uint32_t i = 0; i < 6; i++) {
        if (clipMask_ & (1u << i)) device_->SetClipPlane(i, clipPlanes_[i]);
    }
    for (uint32_t i = 0; i < 256; i++) if (vsConstFMask_[i]) device_->SetVertexShaderConstantF(i, vsConstF_[i], 1);
    for (uint32_t i = 0; i < 16; i++) {
        if (vsConstIMask_ & (1u << i)) device_->SetVertexShaderConstantI(i, vsConstI_[i], 1);
        if (vsConstBMask_ & (1u << i)) device_->SetVertexShaderConstantB(i, &vsConstB_[i], 1);
        if (psConstIMask_ & (1u << i)) device_->SetPixelShaderConstantI(i, psConstI_[i], 1);
        if (psConstBMask_ & (1u << i)) device_->SetPixelShaderConstantB(i, &psConstB_[i], 1);
    }
    for (uint32_t i = 0; i < 224; i++) if (psConstFMask_[i]) device_->SetPixelShaderConstantF(i, psConstF_[i], 1);
    return D3D_S_OK;
}

void MtlStateBlock9::recordRenderState(uint32_t state, uint32_t val) {
    for (auto &kv : rs_) { if (kv.first == state) { kv.second = val; return; } }
    rs_.push_back({state, val});
}
void MtlStateBlock9::recordSamplerState(uint32_t slot, uint32_t type, uint32_t val) {
    uint32_t k = (slot << 8) | type;
    for (auto &kv : ss_) { if (kv.first == k) { kv.second = val; return; } }
    ss_.push_back({k, val});
}
void MtlStateBlock9::recordTexStageState(uint32_t stage, uint32_t type, uint32_t val) {
    uint32_t k = (stage << 8) | type;
    for (auto &kv : tss_) { if (kv.first == k) { kv.second = val; return; } }
    tss_.push_back({k, val});
}
void MtlStateBlock9::recordTexture(uint32_t slot, IMtlDirect3DBaseTexture9 *tex) {
    if (slot >= kMaxSamplers) return;
    if (tex) tex->AddRef();
    if (textures_[slot]) textures_[slot]->Release();
    textures_[slot] = tex;
    texMask_ |= (1u << slot);
}
void MtlStateBlock9::recordVs(MtlVertexShader9 *vs) {
    if (vs) vs->AddRef();
    if (vs_) vs_->Release();
    vs_ = vs; hasVs_ = true;
}
void MtlStateBlock9::recordPs(MtlPixelShader9 *ps) {
    if (ps) ps->AddRef();
    if (ps_) ps_->Release();
    ps_ = ps; hasPs_ = true;
}
void MtlStateBlock9::recordVdecl(MtlVertexDeclaration9 *decl) {
    if (decl) decl->AddRef();
    if (vdecl_) vdecl_->Release();
    vdecl_ = decl; hasDecl_ = true;
}
void MtlStateBlock9::recordFvf(uint32_t fvf) { fvf_ = fvf; hasFvf_ = true; }
void MtlStateBlock9::recordStream(uint32_t idx, MtlVertexBuffer9 *vb, uint32_t offset, uint32_t stride) {
    if (idx >= 16) return;
    if (vb) vb->AddRef();
    if (streams_[idx].vb) streams_[idx].vb->Release();
    streams_[idx].vb = vb; streams_[idx].offset = offset; streams_[idx].stride = stride;
    streamMask_ |= (1u << idx);
}
void MtlStateBlock9::recordStreamFreq(uint32_t idx, uint32_t freq) {
    if (idx >= 16) return;
    streams_[idx].freq = freq;
    streamFreqMask_ |= (1u << idx);
}
void MtlStateBlock9::recordIndices(MtlIndexBuffer9 *ib) {
    if (ib) ib->AddRef();
    if (ib_) ib_->Release();
    ib_ = ib; hasIb_ = true;
}
void MtlStateBlock9::recordViewport(const D3DMtlViewport &vp) { vp_ = vp; hasVp_ = true; }
void MtlStateBlock9::recordScissor(const D3DMtlRect &r) { scissor_ = r; hasScissor_ = true; }
void MtlStateBlock9::recordClipPlane(uint32_t idx, const float p[4]) {
    if (idx >= 6) return;
    std::memcpy(clipPlanes_[idx], p, 16);
    clipMask_ |= (1u << idx);
}
void MtlStateBlock9::recordVsConstF(uint32_t start, const float *data, uint32_t count) {
    for (uint32_t i = 0; i < count && start + i < 256; i++) {
        std::memcpy(vsConstF_[start + i], data + i * 4, 16);
        vsConstFMask_[start + i] = 1;
    }
}
void MtlStateBlock9::recordVsConstI(uint32_t start, const int32_t *data, uint32_t count) {
    for (uint32_t i = 0; i < count && start + i < 16; i++) {
        std::memcpy(vsConstI_[start + i], data + i * 4, 16);
        vsConstIMask_ |= uint16_t(1u << (start + i));
    }
}
void MtlStateBlock9::recordVsConstB(uint32_t start, const uint32_t *data, uint32_t count) {
    for (uint32_t i = 0; i < count && start + i < 16; i++) {
        vsConstB_[start + i] = data[i] ? 1u : 0u;
        vsConstBMask_ |= uint16_t(1u << (start + i));
    }
}
void MtlStateBlock9::recordPsConstF(uint32_t start, const float *data, uint32_t count) {
    for (uint32_t i = 0; i < count && start + i < 224; i++) {
        std::memcpy(psConstF_[start + i], data + i * 4, 16);
        psConstFMask_[start + i] = 1;
    }
}
void MtlStateBlock9::recordPsConstI(uint32_t start, const int32_t *data, uint32_t count) {
    for (uint32_t i = 0; i < count && start + i < 16; i++) {
        std::memcpy(psConstI_[start + i], data + i * 4, 16);
        psConstIMask_ |= uint16_t(1u << (start + i));
    }
}
void MtlStateBlock9::recordPsConstB(uint32_t start, const uint32_t *data, uint32_t count) {
    for (uint32_t i = 0; i < count && start + i < 16; i++) {
        psConstB_[start + i] = data[i] ? 1u : 0u;
        psConstBMask_ |= uint16_t(1u << (start + i));
    }
}

/* ========================================================================= */
/* MtlSwapChain9 + CAMetalLayer                                              */
/* ========================================================================= */

struct MetalViewAttachReq {
    SDL_Window   *win;
    id<MTLDevice> dev;
    bool          vsync;
    uint32_t      bbW;
    uint32_t      bbH;
    SDL_MetalView outView;
    CAMetalLayer *outLayer;
};

static CGSize computeTargetDrawableSize(SDL_Window *win, uint32_t bbW, uint32_t bbH) {
    int pw = 0, ph = 0;
    if (!win || !SDL_GetWindowSizeInPixels(win, &pw, &ph) || pw <= 0 || ph <= 0) {
        return CGSizeMake(std::max(bbW, 1u), std::max(bbH, 1u));
    }
    /* Avoid Retina 5K/8K drawable supersampling above backbuffer resolution; CA scales cleanly in HW */
    if (bbW > 0 && bbH > 0 && uint32_t(pw) > bbW && uint32_t(ph) > bbH) {
        return CGSizeMake(bbW, bbH);
    }
    return CGSizeMake(pw, ph);
}

static void attachMetalViewOnMain(void *userdata) {
    auto *r = static_cast<MetalViewAttachReq *>(userdata);
    if (!r->win) return;
    r->outView = SDL_Metal_CreateView(r->win);
    if (!r->outView) {
        fprintf(stderr, "RunnerMetal: SDL_Metal_CreateView failed: %s\n", SDL_GetError());
        return;
    }
    CAMetalLayer *layer = (__bridge CAMetalLayer *)SDL_Metal_GetLayer(r->outView);
    if (layer) {
        layer.device = r->dev;
        layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
        // Allow direct blit copyFromTexture on 1:1 Present instead of forcing a shader blit pass.
        layer.framebufferOnly = NO;
        layer.maximumDrawableCount = 3;
        layer.displaySyncEnabled = r->vsync ? YES : NO;
        layer.contentsGravity = kCAGravityResizeAspect;
        layer.backgroundColor = [NSColor blackColor].CGColor;
        layer.drawableSize = computeTargetDrawableSize(r->win, r->bbW, r->bbH);
        r->outLayer = layer;
    }
}

static void destroyMetalViewOnMain(void *userdata) {
    SDL_MetalView view = static_cast<SDL_MetalView>(userdata);
    if (view) SDL_Metal_DestroyView(view);
}

struct MetalLayerSyncReq {
    SDL_Window   *win;
    CAMetalLayer *layer;
    bool          vsync;
    uint32_t      bbW;
    uint32_t      bbH;
};

static void syncMetalLayerOnMain(void *userdata) {
    auto *r = static_cast<MetalLayerSyncReq *>(userdata);
    if (!r->layer) return;
    r->layer.displaySyncEnabled = r->vsync ? YES : NO;
    if (r->win) {
        CGSize target = computeTargetDrawableSize(r->win, r->bbW, r->bbH);
        CGSize cur = r->layer.drawableSize;
        if (int(cur.width) != int(target.width) || int(cur.height) != int(target.height)) {
            r->layer.drawableSize = target;
        }
    }
}

MtlSwapChain9::MtlSwapChain9(MtlDirect3DDevice9 *dev, const D3DMtlPresentParameters &pp)
    : device_(dev), pp_(pp) {}

MtlSwapChain9::~MtlSwapChain9() {
    if (backBuffer_) {
        backBuffer_->Release();
        backBuffer_ = nullptr;
    }
    detachLayer();
}

void MtlSwapChain9::attachLayerToWindow(SDL_Window *win) {
    if (!win) return;
    if (sdlWindow_ == win && metalLayer_ != nil) return;
    detachLayer();
    sdlWindow_ = win;
    MetalViewAttachReq req = {};
    req.win = win;
    req.dev = device_->mtlDevice();
    req.vsync = (pp_.PresentationInterval != D3DM_PRESENT_INTERVAL_IMMEDIATE);
    req.bbW = pp_.BackBufferWidth;
    req.bbH = pp_.BackBufferHeight;
    SDL_RunOnMainThread(attachMetalViewOnMain, &req, true);
    metalView_ = req.outView;
    metalLayer_ = req.outLayer;
}

void MtlSwapChain9::detachLayer() {
    metalLayer_ = nil;
    if (metalView_) {
        SDL_RunOnMainThread(destroyMetalViewOnMain, metalView_, true);
        metalView_ = nullptr;
    }
    sdlWindow_ = nullptr;
}

void MtlSwapChain9::updateDrawableSize() {
    if (!metalLayer_ || !sdlWindow_) return;
    CGSize target = computeTargetDrawableSize(sdlWindow_, pp_.BackBufferWidth, pp_.BackBufferHeight);
    CGSize cur = metalLayer_.drawableSize;
    bool wantVsync = (pp_.PresentationInterval != D3DM_PRESENT_INTERVAL_IMMEDIATE);
    if (int(cur.width) != int(target.width) || int(cur.height) != int(target.height) ||
        (metalLayer_.displaySyncEnabled != (wantVsync ? YES : NO))) {
        MetalLayerSyncReq req = { sdlWindow_, metalLayer_, wantVsync, pp_.BackBufferWidth, pp_.BackBufferHeight };
        SDL_RunOnMainThread(syncMetalLayerOnMain, &req, true);
    }
}

bool MtlSwapChain9::init(const D3DMtlPresentParameters &pp) {
    pp_ = pp;
    if (pp_.BackBufferWidth == 0) pp_.BackBufferWidth = 1280;
    if (pp_.BackBufferHeight == 0) pp_.BackBufferHeight = 720;
    if (pp_.BackBufferFormat == 0 || pp_.BackBufferFormat == D3DM_FMT_UNKNOWN) {
        pp_.BackBufferFormat = D3DM_FMT_X8R8G8B8;
    }
    if (pp_.BackBufferCount == 0) pp_.BackBufferCount = 1;

    SDL_Window *win = static_cast<SDL_Window *>(pp_.hDeviceWindow);
    if (win) attachLayerToWindow(win);

    /* Internal offscreen backbuffer texture: keep PixelFormatView off on Apple GPU for lossless compression */
    MtlFormatInfo fi = mapD3dFormat(pp_.BackBufferFormat);
    if (fi.pixelFormat == MTLPixelFormatInvalid) fi = mapD3dFormat(D3DM_FMT_A8R8G8B8);

    MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fi.pixelFormat
                                                                                  width:pp_.BackBufferWidth
                                                                                 height:pp_.BackBufferHeight
                                                                              mipmapped:NO];
    td.storageMode = MTLStorageModePrivate;
    MTLTextureUsage bbUsage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    if (!device_->isAppleGpu()) bbUsage |= MTLTextureUsagePixelFormatView;
    td.usage = bbUsage;
    id<MTLTexture> bbTex = [device_->mtlDevice() newTextureWithDescriptor:td];
    if (!bbTex) return false;

    if (!backBuffer_) {
        backBuffer_ = new MtlSurface9(device_, pp_.BackBufferWidth, pp_.BackBufferHeight,
                                      pp_.BackBufferFormat, D3DM_USAGE_RENDERTARGET, D3DM_POOL_DEFAULT,
                                      0, 0, bbTex, nullptr, 0, 0);
    } else {
        backBuffer_->replaceTexture(bbTex, pp_.BackBufferWidth, pp_.BackBufferHeight, pp_.BackBufferFormat);
    }
    return true;
}

bool MtlSwapChain9::resize(const D3DMtlPresentParameters &pp) {
    D3DMtlPresentParameters next = pp;
    if (!next.hDeviceWindow) next.hDeviceWindow = pp_.hDeviceWindow;
    bool ok = init(next);
    updateDrawableSize();
    return ok;
}

MTL_HRESULT MtlSwapChain9::Present(const D3DMtlRect *pSourceRect, const D3DMtlRect *pDestRect,
                                   void *hDestWindowOverride, const void *pDirtyRegion, uint32_t dwFlags) {
    (void)hDestWindowOverride;
    (void)pDirtyRegion;
    updateDrawableSize();
    return device_->presentSwapChain(this, pSourceRect, pDestRect, dwFlags);
}

MTL_HRESULT MtlSwapChain9::GetFrontBufferData(IMtlDirect3DSurface9 *pDestSurface) {
    return device_->GetRenderTargetData(backBuffer_, pDestSurface);
}

MTL_HRESULT MtlSwapChain9::GetBackBuffer(uint32_t iBackBuffer, uint32_t Type, IMtlDirect3DSurface9 **ppBackBuffer) {
    (void)Type;
    if (!ppBackBuffer || iBackBuffer > 0 || !backBuffer_) return D3D_ERR_INVALIDCALL;
    backBuffer_->AddRef();
    *ppBackBuffer = backBuffer_;
    return D3D_S_OK;
}

MTL_HRESULT MtlSwapChain9::GetRasterStatus(D3DMtlRasterStatus *pRasterStatus) {
    if (!pRasterStatus) return D3D_ERR_INVALIDCALL;
    pRasterStatus->InVBlank = 1;
    pRasterStatus->ScanLine = 0;
    return D3D_S_OK;
}

MTL_HRESULT MtlSwapChain9::GetDisplayMode(D3DMtlDisplayMode *pMode) {
    if (!pMode) return D3D_ERR_INVALIDCALL;
    pMode->Width = pp_.BackBufferWidth;
    pMode->Height = pp_.BackBufferHeight;
    pMode->RefreshRate = 60;
    pMode->Format = pp_.BackBufferFormat;
    return D3D_S_OK;
}

MTL_HRESULT MtlSwapChain9::GetDevice(IMtlDirect3DDevice9 **ppDevice) {
    if (!ppDevice) return D3D_ERR_INVALIDCALL;
    device_->AddRef();
    *ppDevice = device_;
    return D3D_S_OK;
}

MTL_HRESULT MtlSwapChain9::GetPresentParameters(D3DMtlPresentParameters *pPP) {
    if (!pPP) return D3D_ERR_INVALIDCALL;
    *pPP = pp_;
    return D3D_S_OK;
}

/* ========================================================================= */
/* MtlDirect3DDevice9                                                        */
/* ========================================================================= */

MtlDirect3DDevice9::MtlDirect3DDevice9(IMtlDirect3D9 *d3d9, id<MTLDevice> mtlDevice,
                                       uint32_t adapter, uint32_t devType, void *focusWindow,
                                       uint32_t behaviorFlags, const D3DMtlPresentParameters &pp)
    : d3d9_(d3d9), mtlDevice_(mtlDevice), presentParams_(pp) {
    if (d3d9_) d3d9_->AddRef();
    creationParams_.AdapterOrdinal = adapter;
    creationParams_.DeviceType     = devType;
    creationParams_.hFocusWindow   = focusWindow;
    creationParams_.BehaviorFlags  = behaviorFlags;

    if (!presentParams_.hDeviceWindow) presentParams_.hDeviceWindow = focusWindow;

    for (uint32_t i = 0; i < 256; i++) {
        uint16_t v = uint16_t(i << 8);
        gammaRamp_.red[i] = gammaRamp_.green[i] = gammaRamp_.blue[i] = v;
    }

    isAppleGpu_ = [mtlDevice_ supportsFamily:MTLGPUFamilyApple7];
    const char *perfEnv = getenv("ME_PC_PERF");
    perfLogEnabled_ = (perfEnv && *perfEnv == '1');
    const char *passEnv = getenv("ME_PC_GPU_PASSES");
    gpuPassLogEnabled_ = (passEnv && *passEnv == '1');
    const char *stallEnv = getenv("ME_PC_LOG_STALLS");
    stallLogEnabled_ = (stallEnv && *stallEnv == '1');
}

MtlDirect3DDevice9::~MtlDirect3DDevice9() {
    unregisterWarmDevice(this);
    warmStop_.store(true, std::memory_order_release);
    if (shaderCompiler_) shaderCompiler_->shutdownWorkers();

    flushCommands(true);
    waitForCommandSeq(lastCommittedSeq_.load(std::memory_order_acquire), 5000);

    for (MtlSurface9 *s : pendingClearSurfaces_) {
        if (s) {
            s->clearAllPendingClears();
            s->setInPendingClearList(false);
        }
    }
    pendingClearSurfaces_.clear();

    if (recordingBlock_) { recordingBlock_->Release(); recordingBlock_ = nullptr; }
    state_.releaseRefs();

    for (auto &kv : fvfDeclCache_) {
        if (kv.second) kv.second->Release();
    }
    fvfDeclCache_.clear();

    for (int i = 0; i < 4; i++) {
        if (renderTargets_[i]) { renderTargets_[i]->Release(); renderTargets_[i] = nullptr; }
    }
    if (depthStencil_) { depthStencil_->Release(); depthStencil_ = nullptr; }
    if (autoDepthStencil_) { autoDepthStencil_->Release(); autoDepthStencil_ = nullptr; }
    if (primarySwapChain_) { primarySwapChain_->Release(); primarySwapChain_ = nullptr; }

    /* Balance initial frame semaphore permit before releasing */
    if (frameSem_) {
        dispatch_semaphore_signal(frameSem_);
        frameSem_ = nullptr;
    }

    if (sharedBufferPool_) {
        sharedBufferPool_->shutdown();
        sharedBufferPool_.reset();
    }

    if (d3d9_) { d3d9_->Release(); d3d9_ = nullptr; }
}

bool MtlDirect3DDevice9::initialize() {
    cmdQueue_ = [mtlDevice_ newCommandQueue];
    if (!cmdQueue_) return false;

    ringBuffer_       = std::make_unique<MtlRingBuffer>(mtlDevice_);
    sharedBufferPool_ = std::make_shared<MtlSharedBufferPool>(mtlDevice_, &completedCmdSeq_);
    shaderCompiler_   = std::make_unique<MtlShaderCompiler>(mtlDevice_);
    pipelineCache_    = std::make_unique<MtlPipelineCache>(mtlDevice_);
    pipelineCache_->seedCompilerFromManifest(*shaderCompiler_);

    visBuffer_ = [mtlDevice_ newBufferWithLength:kMaxVisibilitySlots * sizeof(uint64_t)
                                         options:MTLResourceStorageModeShared];
    if (visBuffer_) {
        visPtr_ = static_cast<uint64_t *>([visBuffer_ contents]);
        std::memset(visPtr_, 0, kMaxVisibilitySlots * sizeof(uint64_t));
    }

    frameSem_ = dispatch_semaphore_create(MtlRingBuffer::kFrameSlots);
    dispatch_semaphore_wait(frameSem_, DISPATCH_TIME_FOREVER);
    ringBuffer_->beginFrame(0);

    primarySwapChain_ = new MtlSwapChain9(this, presentParams_);
    if (!primarySwapChain_->init(presentParams_)) return false;
    presentParams_ = primarySwapChain_->presentParams();

    /* Bind backbuffer to RT0 */
    renderTargets_[0] = primarySwapChain_->backBuffer();
    if (renderTargets_[0]) renderTargets_[0]->AddRef();

    /* Auto depth-stencil */
    if (presentParams_.EnableAutoDepthStencil) {
        uint32_t dsFmt = presentParams_.AutoDepthStencilFormat ? presentParams_.AutoDepthStencilFormat : D3DM_FMT_D24S8;
        autoDepthStencil_ = new MtlSurface9(this, presentParams_.BackBufferWidth, presentParams_.BackBufferHeight,
                                            dsFmt, D3DM_USAGE_DEPTHSTENCIL, D3DM_POOL_DEFAULT, 0, 0);
        depthStencil_ = autoDepthStencil_;
        depthStencil_->AddRef();
    }

    state_.resetDefaults(presentParams_.BackBufferWidth, presentParams_.BackBufferHeight);
    vsConstFVersion_++;
    psConstFVersion_++;
    if (!presentParams_.EnableAutoDepthStencil) {
        state_.renderStates[D3DM_RS_ZENABLE] = 0;
    }

    fprintf(stderr, "RunnerMetal: IDirect3DDevice9 created (%ux%u fmt=%u autoDS=%u/%u appleGPU=%d)\n",
            presentParams_.BackBufferWidth, presentParams_.BackBufferHeight,
            presentParams_.BackBufferFormat, presentParams_.EnableAutoDepthStencil,
            presentParams_.AutoDepthStencilFormat, isAppleGpu_ ? 1 : 0);
    registerWarmDevice(this);
    return true;
}

void MtlDirect3DDevice9::onWarmSetMap(const char *name) {
    if (!name || !*name) return;
    std::string mapName;
    for (const unsigned char *p = reinterpret_cast<const unsigned char *>(name); *p; ++p) {
        mapName.push_back(char(std::tolower(*p)));
    }
    uint64_t mapHash = hashMapStem(mapName.c_str());
    if (mapHash == 0) return;

    uint64_t prevHash = currentMapHash_.exchange(mapHash, std::memory_order_acq_rel);
    if (prevHash == mapHash) return;
    if (!pipelineCache_ || !shaderCompiler_ || warmStop_.load(std::memory_order_acquire)) return;

    auto enqueueBatch = [&](std::string label, std::vector<MtlPsoManifestRecord> recs, bool highPrio, uint64_t genCheck) {
        if (recs.empty()) {
            if (highPrio && stallLogEnabled_) {
                fprintf(stderr, "STALL_LOG: warmMap '%s' (0 manifest records)\n", label.c_str());
            }
            return;
        }
        struct WarmBatch {
            std::string mapName;
            std::vector<MtlPsoManifestRecord> records;
            uint64_t gen = 0;
            bool checkGen = false;
            std::chrono::steady_clock::time_point tStart;
            std::atomic<size_t> cursor{0};
            std::atomic<uint32_t> warmed{0};
            std::atomic<uint32_t> alreadyHot{0};
            std::atomic<size_t> remainingTasks{0};
        };

        auto batch = std::make_shared<WarmBatch>();
        batch->mapName  = std::move(label);
        batch->records  = std::move(recs);
        batch->gen      = genCheck;
        batch->checkGen = highPrio;
        batch->tStart   = std::chrono::steady_clock::now();

        size_t taskCount = std::min<size_t>(std::max<size_t>(shaderCompiler_->workerCount(), 1u),
                                            batch->records.size());
        batch->remainingTasks.store(taskCount, std::memory_order_release);

        for (size_t t = 0; t < taskCount; t++) {
            shaderCompiler_->enqueueWork([this, batch]() {
                shaderCompiler_->waitForPrewarmDone();
                while (!warmStop_.load(std::memory_order_acquire) &&
                       (!batch->checkGen || activeWarmGen_.load(std::memory_order_acquire) == batch->gen)) {
                    size_t idx = batch->cursor.fetch_add(1u, std::memory_order_relaxed);
                    if (idx >= batch->records.size()) break;
                    const MtlPsoManifestRecord &rec = batch->records[idx];
                    auto vs = shaderCompiler_->getOrLoadParsedByHash(rec.vsBytecodeHash, true, true);
                    if (!vs) continue;
                    const MtlCompiledShader *psCompiled = nullptr;
                    if (rec.psBytecodeHash != 0) {
                        auto ps = shaderCompiler_->getOrLoadParsedByHash(rec.psBytecodeHash, false, true);
                        if (!ps) continue;
                        bool hasDepth = (rec.psKey.flags & 1u) != 0u;
                        bool flat = (rec.psKey.flags & 2u) != 0u;
                        bool alphaTest = (rec.psKey.flags & 4u) != 0u;
                        bool ptSprite = (rec.psKey.flags & 8u) != 0u;
                        psCompiled = shaderCompiler_->getOrCompilePsVariant(
                            *ps, vs.get(),
                            rec.psKey.colorWriteMask, hasDepth, flat, alphaTest, ptSprite,
                            rec.psKey.samplerDepthCmpMask, rec.psKey.samplerNullMask,
                            rec.psKey.samplerProjMask, rec.psKey.samplerGatherMask,
                            /*suppressStallHook=*/true);
                        if (!psCompiled) continue;
                    }
                    MtlStreamBindingInfo streams[16] = {};
                    for (uint32_t s = 0; s < 16u; s++) {
                        streams[s].bound       = (rec.streamBoundMask & uint16_t(1u << s)) != 0u;
                        streams[s].stride      = rec.streamStrides[s];
                        streams[s].freqSetting = rec.streamFreqs[s];
                    }
                    if (pipelineCache_->warmRenderPipeline(
                            rec.psoKey, *vs, psCompiled, rec.elems, rec.elemCount, streams)) {
                        batch->warmed.fetch_add(1u, std::memory_order_relaxed);
                    } else {
                        batch->alreadyHot.fetch_add(1u, std::memory_order_relaxed);
                    }
                }
                if (batch->remainingTasks.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
                    if (stallLogEnabled_ && !warmStop_.load(std::memory_order_acquire)) {
                        double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - batch->tStart).count();
                        fprintf(stderr, "STALL_LOG: warmMap '%s' (%zu records: %u warmed, %u alreadyHot) in %.1f ms\n",
                                batch->mapName.c_str(), batch->records.size(),
                                batch->warmed.load(), batch->alreadyHot.load(), ms);
                    }
                }
            }, highPrio);
        }
    };

    uint64_t gen = activeWarmGen_.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    enqueueBatch(mapName, pipelineCache_->getManifestRecordsForMap(mapHash), true, gen);

    static const bool prewarmAll = ([] {
        const char *e = getenv("ME_PC_PREWARM_ALL_MAPS");
        return e && *e == '1';
    })();
    if (prewarmAll && !allMapsWarmed_.exchange(true, std::memory_order_acq_rel)) {
        auto rest = pipelineCache_->getAllManifestRecordsExceptMap(mapHash);
        if (!rest.empty()) {
            enqueueBatch("all_other_maps", std::move(rest), false, 0);
        }
    }
}

uint32_t MtlDirect3DDevice9::GetAvailableTextureMem() {
    uint64_t mem = [mtlDevice_ recommendedMaxWorkingSetSize];
    if (mem == 0) mem = 2048ull * 1024ull * 1024ull;
    if (mem > 0x7FF00000ull) mem = 0x7FF00000ull; /* Keep signed 32-bit positive for UE3 */
    return uint32_t(mem);
}

MTL_HRESULT MtlDirect3DDevice9::GetDirect3D(IMtlDirect3D9 **ppD3D9) {
    if (!ppD3D9 || !d3d9_) return D3D_ERR_INVALIDCALL;
    d3d9_->AddRef();
    *ppD3D9 = d3d9_;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetDeviceCaps(D3DMtlCaps9 *pCaps) {
    return d3d9_->GetDeviceCaps(creationParams_.AdapterOrdinal, creationParams_.DeviceType, pCaps);
}

MTL_HRESULT MtlDirect3DDevice9::GetDisplayMode(uint32_t iSwapChain, D3DMtlDisplayMode *pMode) {
    if (iSwapChain != 0 || !primarySwapChain_) return D3D_ERR_INVALIDCALL;
    return primarySwapChain_->GetDisplayMode(pMode);
}

MTL_HRESULT MtlDirect3DDevice9::GetCreationParameters(D3DMtlCreationParameters *pParameters) {
    if (!pParameters) return D3D_ERR_INVALIDCALL;
    *pParameters = creationParams_;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateAdditionalSwapChain(D3DMtlPresentParameters *pPP, IMtlDirect3DSwapChain9 **pSwapChain) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pPP || !pSwapChain) return D3D_ERR_INVALIDCALL;
    auto *sc = new MtlSwapChain9(this, *pPP);
    if (!sc->init(*pPP)) {
        sc->Release();
        return D3D_ERR_INVALIDCALL;
    }
    *pPP = sc->presentParams();
    *pSwapChain = sc;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetSwapChain(uint32_t iSwapChain, IMtlDirect3DSwapChain9 **pSwapChain) {
    if (iSwapChain != 0 || !pSwapChain || !primarySwapChain_) return D3D_ERR_INVALIDCALL;
    primarySwapChain_->AddRef();
    *pSwapChain = primarySwapChain_;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::Reset(D3DMtlPresentParameters *pPP) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pPP) return D3D_ERR_INVALIDCALL;

    flushCommands(true);
    for (MtlSurface9 *s : pendingClearSurfaces_) {
        if (s) {
            s->clearAllPendingClears();
            s->setInPendingClearList(false);
        }
    }
    pendingClearSurfaces_.clear();
    pendingColorDontCare_ = false;
    activeVisSlot_ = kInvalidVisSlot;
    invalidatePassBindCache();

    if (!primarySwapChain_->resize(*pPP)) return D3D_ERR_INVALIDCALL;
    presentParams_ = primarySwapChain_->presentParams();
    *pPP = presentParams_;

    for (int i = 0; i < 4; i++) {
        if (renderTargets_[i]) { renderTargets_[i]->Release(); renderTargets_[i] = nullptr; }
    }
    renderTargets_[0] = primarySwapChain_->backBuffer();
    if (renderTargets_[0]) renderTargets_[0]->AddRef();

    if (depthStencil_) { depthStencil_->Release(); depthStencil_ = nullptr; }
    if (autoDepthStencil_) { autoDepthStencil_->Release(); autoDepthStencil_ = nullptr; }
    if (presentParams_.EnableAutoDepthStencil) {
        uint32_t dsFmt = presentParams_.AutoDepthStencilFormat ? presentParams_.AutoDepthStencilFormat : D3DM_FMT_D24S8;
        autoDepthStencil_ = new MtlSurface9(this, presentParams_.BackBufferWidth, presentParams_.BackBufferHeight,
                                            dsFmt, D3DM_USAGE_DEPTHSTENCIL, D3DM_POOL_DEFAULT, 0, 0);
        depthStencil_ = autoDepthStencil_;
        depthStencil_->AddRef();
    }

    state_.resetDefaults(presentParams_.BackBufferWidth, presentParams_.BackBufferHeight);
    vsConstFVersion_++;
    psConstFVersion_++;
    if (!presentParams_.EnableAutoDepthStencil) {
        state_.renderStates[D3DM_RS_ZENABLE] = 0;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::Present(const D3DMtlRect *pSourceRect, const D3DMtlRect *pDestRect,
                                        void *hDestWindowOverride, const void *pDirtyRegion) {
    return primarySwapChain_->Present(pSourceRect, pDestRect, hDestWindowOverride, pDirtyRegion, 0);
}

MTL_HRESULT MtlDirect3DDevice9::GetBackBuffer(uint32_t iSwapChain, uint32_t iBackBuffer, uint32_t Type, IMtlDirect3DSurface9 **ppBackBuffer) {
    if (iSwapChain != 0 || !primarySwapChain_) return D3D_ERR_INVALIDCALL;
    return primarySwapChain_->GetBackBuffer(iBackBuffer, Type, ppBackBuffer);
}

MTL_HRESULT MtlDirect3DDevice9::GetRasterStatus(uint32_t iSwapChain, D3DMtlRasterStatus *pRasterStatus) {
    if (iSwapChain != 0 || !primarySwapChain_) return D3D_ERR_INVALIDCALL;
    return primarySwapChain_->GetRasterStatus(pRasterStatus);
}

void MtlDirect3DDevice9::SetGammaRamp(uint32_t, uint32_t, const D3DMtlGammaRamp *pRamp) {
    if (pRamp) gammaRamp_ = *pRamp;
}

void MtlDirect3DDevice9::GetGammaRamp(uint32_t, D3DMtlGammaRamp *pRamp) {
    if (pRamp) *pRamp = gammaRamp_;
}

/* ========================================================================= */
/* Resource Creation                                                         */
/* ========================================================================= */

MTL_HRESULT MtlDirect3DDevice9::CreateTexture(uint32_t Width, uint32_t Height, uint32_t Levels,
                                              uint32_t Usage, uint32_t Format, uint32_t Pool,
                                              IMtlDirect3DTexture9 **ppTexture, void **pSharedHandle) {
    (void)pSharedHandle;
    if (!ppTexture || Width == 0 || Height == 0) return D3D_ERR_INVALIDCALL;
    MtlFormatInfo fi = mapD3dFormat(Format);
    if (!fi.isNullFormat && fi.pixelFormat == MTLPixelFormatInvalid) return D3D_ERR_NOTAVAILABLE;
    *ppTexture = new MtlTexture9(this, Width, Height, Levels, Usage, Format, Pool);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateVolumeTexture(uint32_t Width, uint32_t Height, uint32_t Depth,
                                                    uint32_t Levels, uint32_t Usage, uint32_t Format, uint32_t Pool,
                                                    IMtlDirect3DVolumeTexture9 **ppVolumeTexture, void **pSharedHandle) {
    (void)pSharedHandle;
    if (!ppVolumeTexture || Width == 0 || Height == 0 || Depth == 0) return D3D_ERR_INVALIDCALL;
    MtlFormatInfo fi = mapD3dFormat(Format);
    if (fi.pixelFormat == MTLPixelFormatInvalid) return D3D_ERR_NOTAVAILABLE;
    *ppVolumeTexture = new MtlVolumeTexture9(this, Width, Height, Depth, Levels, Usage, Format, Pool);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateCubeTexture(uint32_t EdgeLength, uint32_t Levels, uint32_t Usage,
                                                  uint32_t Format, uint32_t Pool,
                                                  IMtlDirect3DCubeTexture9 **ppCubeTexture, void **pSharedHandle) {
    (void)pSharedHandle;
    if (!ppCubeTexture || EdgeLength == 0) return D3D_ERR_INVALIDCALL;
    MtlFormatInfo fi = mapD3dFormat(Format);
    if (fi.pixelFormat == MTLPixelFormatInvalid) return D3D_ERR_NOTAVAILABLE;
    *ppCubeTexture = new MtlCubeTexture9(this, EdgeLength, Levels, Usage, Format, Pool);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateVertexBuffer(uint32_t Length, uint32_t Usage, uint32_t FVF, uint32_t Pool,
                                                   IMtlDirect3DVertexBuffer9 **ppVertexBuffer, void **pSharedHandle) {
    (void)pSharedHandle;
    if (!ppVertexBuffer) return D3D_ERR_INVALIDCALL;
    *ppVertexBuffer = new MtlVertexBuffer9(this, Length, Usage, FVF, Pool);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateIndexBuffer(uint32_t Length, uint32_t Usage, uint32_t Format, uint32_t Pool,
                                                  IMtlDirect3DIndexBuffer9 **ppIndexBuffer, void **pSharedHandle) {
    (void)pSharedHandle;
    if (!ppIndexBuffer) return D3D_ERR_INVALIDCALL;
    *ppIndexBuffer = new MtlIndexBuffer9(this, Length, Usage, Format, Pool);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateRenderTarget(uint32_t Width, uint32_t Height, uint32_t Format,
                                                   uint32_t MultiSample, uint32_t MultisampleQuality,
                                                   uint32_t Lockable, IMtlDirect3DSurface9 **ppSurface, void **pSharedHandle) {
    (void)pSharedHandle; (void)Lockable;
    if (!ppSurface || Width == 0 || Height == 0) return D3D_ERR_INVALIDCALL;
    MtlFormatInfo fi = mapD3dFormat(Format);
    if (!fi.isNullFormat && fi.pixelFormat == MTLPixelFormatInvalid) return D3D_ERR_NOTAVAILABLE;
    *ppSurface = new MtlSurface9(this, Width, Height, Format, D3DM_USAGE_RENDERTARGET, D3DM_POOL_DEFAULT,
                                 MultiSample, MultisampleQuality);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateDepthStencilSurface(uint32_t Width, uint32_t Height, uint32_t Format,
                                                          uint32_t MultiSample, uint32_t MultisampleQuality,
                                                          uint32_t Discard, IMtlDirect3DSurface9 **ppSurface, void **pSharedHandle) {
    (void)pSharedHandle; (void)Discard;
    if (!ppSurface || Width == 0 || Height == 0) return D3D_ERR_INVALIDCALL;
    MtlFormatInfo fi = mapD3dFormat(Format);
    if (!fi.isDepth || fi.pixelFormat == MTLPixelFormatInvalid) return D3D_ERR_NOTAVAILABLE;
    *ppSurface = new MtlSurface9(this, Width, Height, Format, D3DM_USAGE_DEPTHSTENCIL, D3DM_POOL_DEFAULT,
                                 MultiSample, MultisampleQuality);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateOffscreenPlainSurface(uint32_t Width, uint32_t Height, uint32_t Format,
                                                            uint32_t Pool, IMtlDirect3DSurface9 **ppSurface, void **pSharedHandle) {
    (void)pSharedHandle;
    if (!ppSurface || Width == 0 || Height == 0) return D3D_ERR_INVALIDCALL;
    *ppSurface = new MtlSurface9(this, Width, Height, Format, 0, Pool, 0, 0);
    return D3D_S_OK;
}

/* ========================================================================= */
/* Surface / Texture Transfers & Readback                                    */
/* ========================================================================= */

void MtlDirect3DDevice9::enqueueTextureUpload(const PendingTextureUpload &up) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    pendingUploads_.push_back(up);
}

void MtlDirect3DDevice9::enqueueGenerateMipmaps(id<MTLTexture> tex) {
    if (!tex) return;
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    pendingMipGens_.push_back(tex);
}

void MtlDirect3DDevice9::registerPendingClearSurface(MtlSurface9 *s) {
    if (!s || s->inPendingClearList()) return;
    s->setInPendingClearList(true);
    pendingClearSurfaces_.push_back(s);
}

void MtlDirect3DDevice9::unregisterPendingClearSurface(MtlSurface9 *s) {
    if (!s || !s->inPendingClearList()) return;
    auto it = std::find(pendingClearSurfaces_.begin(), pendingClearSurfaces_.end(), s);
    if (it != pendingClearSurfaces_.end()) {
        *it = pendingClearSurfaces_.back();
        pendingClearSurfaces_.pop_back();
    }
    s->setInPendingClearList(false);
}

void MtlDirect3DDevice9::flushSurfacePendingClear(MtlSurface9 *s) {
    if (!s || !s->hasAnyPendingClear()) return;
    bool clrC = s->hasPendingColorClear();
    bool clrZ = s->hasPendingZClear();
    bool clrS = s->hasPendingSClear();
    D3DCOLOR_T colorVal = s->pendingClearColor();
    float zVal = s->pendingClearZ();
    uint32_t sVal = s->pendingClearStencil();
    s->clearAllPendingClears();
    unregisterPendingClearSurface(s);

    if (s->isNull() || s->texture() == nil) return;

    endRenderPass();
    flushPendingBlits();
    ensureCommandBuffer();

    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.renderTargetWidth  = s->desc().Width;
    rpd.renderTargetHeight = s->desc().Height;
    bool anyAttach = false;

    if (s->formatInfo().isDepth) {
        if (clrZ || clrS) {
            rpd.depthAttachment.texture     = s->texture();
            rpd.depthAttachment.level       = s->mipLevel();
            rpd.depthAttachment.slice       = s->slice();
            rpd.depthAttachment.loadAction  = clrZ ? MTLLoadActionClear : MTLLoadActionLoad;
            rpd.depthAttachment.storeAction = MTLStoreActionStore;
            rpd.depthAttachment.clearDepth  = double(zVal);
            if (s->formatInfo().hasStencil) {
                rpd.stencilAttachment.texture      = s->texture();
                rpd.stencilAttachment.level        = s->mipLevel();
                rpd.stencilAttachment.slice        = s->slice();
                rpd.stencilAttachment.loadAction   = clrS ? MTLLoadActionClear : MTLLoadActionLoad;
                rpd.stencilAttachment.storeAction  = MTLStoreActionStore;
                rpd.stencilAttachment.clearStencil = sVal & 0xFFu;
            }
            anyAttach = true;
        }
    } else if (clrC) {
        rpd.colorAttachments[0].texture     = s->texture();
        rpd.colorAttachments[0].level       = s->mipLevel();
        rpd.colorAttachments[0].slice       = s->slice();
        rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpd.colorAttachments[0].clearColor  = MTLClearColorMake(
            double((colorVal >> 16) & 0xFFu) / 255.0,
            double((colorVal >>  8) & 0xFFu) / 255.0,
            double((colorVal >>  0) & 0xFFu) / 255.0,
            double((colorVal >> 24) & 0xFFu) / 255.0);
        anyAttach = true;
    }

    if (anyAttach) {
        id<MTLRenderCommandEncoder> enc = [cmdBuf_ renderCommandEncoderWithDescriptor:rpd];
        if (enc) [enc endEncoding];
        hasPendingWork_ = true;
    }
}

void MtlDirect3DDevice9::flushPendingClearsForSampledTextures(uint32_t vsSampMask, uint32_t psSampMask) {
    if (pendingClearSurfaces_.empty()) return;

    id<MTLTexture> sampled[20] = {};
    uint32_t sampledCount = 0;
    uint32_t vm = vsSampMask & 0xFu;
    while (vm) {
        uint32_t s = uint32_t(__builtin_ctz(vm));
        vm &= vm - 1u;
        MtlTextureCommon *tc = getTextureCommon(state_.textures[16u + s]);
        if (tc && tc->baseTexture != nil) sampled[sampledCount++] = tc->baseTexture;
    }
    uint32_t pm = psSampMask & 0xFFFFu;
    while (pm) {
        uint32_t s = uint32_t(__builtin_ctz(pm));
        pm &= pm - 1u;
        MtlTextureCommon *tc = getTextureCommon(state_.textures[s]);
        if (tc && tc->baseTexture != nil) sampled[sampledCount++] = tc->baseTexture;
    }
    if (sampledCount == 0) return;

    std::vector<MtlSurface9 *> toFlush;
    for (MtlSurface9 *surf : pendingClearSurfaces_) {
        if (!surf || !surf->hasAnyPendingClear() || surf->texture() == nil) continue;
        id<MTLTexture> stex = surf->texture();
        for (uint32_t i = 0; i < sampledCount; i++) {
            if (sampled[i] == stex) {
                toFlush.push_back(surf);
                break;
            }
        }
    }
    for (MtlSurface9 *surf : toFlush) {
        flushSurfacePendingClear(surf);
    }
}

void MtlDirect3DDevice9::readbackTextureToCpu(id<MTLTexture> tex, uint32_t level, uint32_t slice,
                                              uint32_t d3dFormat, uint32_t w, uint32_t h,
                                              std::vector<uint8_t> &out) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!tex) return;

    if (!pendingClearSurfaces_.empty()) {
        std::vector<MtlSurface9 *> toFlush;
        for (MtlSurface9 *surf : pendingClearSurfaces_) {
            if (surf && surf->hasAnyPendingClear() && surf->texture() == tex) {
                toFlush.push_back(surf);
            }
        }
        for (MtlSurface9 *surf : toFlush) {
            flushSurfacePendingClear(surf);
        }
    }
    endRenderPass();
    flushPendingBlits();
    ensureCommandBuffer();

    uint32_t rowPitch = calcRowPitch(d3dFormat, w);
    uint32_t sliceSize = calcSliceSize(d3dFormat, w, h);
    if (out.size() < sliceSize) out.resize(sliceSize + rowPitch, 0);

    id<MTLBuffer> readBuf = [mtlDevice_ newBufferWithLength:std::max(sliceSize, 16u)
                                                    options:MTLResourceStorageModeShared];
    if (!readBuf) return;

    id<MTLBlitCommandEncoder> blit = [cmdBuf_ blitCommandEncoder];
    [blit copyFromTexture:tex
              sourceSlice:slice
              sourceLevel:level
             sourceOrigin:MTLOriginMake(0, 0, 0)
               sourceSize:MTLSizeMake(w, h, 1)
                 toBuffer:readBuf
        destinationOffset:0
   destinationBytesPerRow:rowPitch
 destinationBytesPerImage:sliceSize];
    [blit endEncoding];
    hasPendingWork_ = true;

    commitCommandBuffer(true, false, nil, false);
    std::memcpy(out.data(), [readBuf contents], sliceSize);
}

static bool validateSurfaceRect(const D3DMtlRect *r, uint32_t fullW, uint32_t fullH,
                                uint32_t &ox, uint32_t &oy, uint32_t &ow, uint32_t &oh) {
    if (!r) {
        ox = 0; oy = 0; ow = fullW; oh = fullH;
        return true;
    }
    if (r->left < 0 || r->top < 0 || r->right < r->left || r->bottom < r->top ||
        uint32_t(r->right) > fullW || uint32_t(r->bottom) > fullH) {
        return false;
    }
    ox = uint32_t(r->left);
    oy = uint32_t(r->top);
    ow = uint32_t(r->right - r->left);
    oh = uint32_t(r->bottom - r->top);
    return true;
}

MTL_HRESULT MtlDirect3DDevice9::UpdateSurface(IMtlDirect3DSurface9 *pSourceSurface, const D3DMtlRect *pSourceRect,
                                              IMtlDirect3DSurface9 *pDestinationSurface, const D3DMtlPoint *pDestPoint) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pSourceSurface || !pDestinationSurface) return D3D_ERR_INVALIDCALL;
    auto *src = static_cast<MtlSurface9 *>(pSourceSurface);
    auto *dst = static_cast<MtlSurface9 *>(pDestinationSurface);

    uint32_t sx = 0, sy = 0, sw = 0, sh = 0;
    if (!validateSurfaceRect(pSourceRect, src->desc().Width, src->desc().Height, sx, sy, sw, sh)) {
        return D3D_ERR_INVALIDCALL;
    }
    if (pDestPoint && (pDestPoint->x < 0 || pDestPoint->y < 0)) return D3D_ERR_INVALIDCALL;
    uint32_t dx = pDestPoint ? uint32_t(pDestPoint->x) : 0u;
    uint32_t dy = pDestPoint ? uint32_t(pDestPoint->y) : 0u;
    if (sw == 0 || sh == 0) return D3D_S_OK;
    if (dx + sw > dst->desc().Width || dy + sh > dst->desc().Height) return D3D_ERR_INVALIDCALL;

    if (!src->cpuData().empty()) {
        if (dst->hasAnyPendingClear()) {
            if (!dst->formatInfo().isDepth && dx == 0 && dy == 0 &&
                sw == dst->desc().Width && sh == dst->desc().Height) {
                dst->clearPendingColorClear();
                if (!dst->hasAnyPendingClear()) unregisterPendingClearSurface(dst);
            } else {
                flushSurfacePendingClear(dst);
            }
        }
        dst->ensureCpuBacking();
        const MtlFormatInfo &fi = dst->formatInfo();
        uint32_t bw = std::max(fi.blockWidth, 1u);
        uint32_t bh = std::max(fi.blockHeight, 1u);
        uint32_t bb = std::max(fi.blockBytes, 1u);
        uint32_t srcPitch = calcRowPitch(src->desc().Format, src->desc().Width);
        uint32_t dstPitch = calcRowPitch(dst->desc().Format, dst->desc().Width);
        uint32_t copyPitch = calcRowPitch(dst->desc().Format, sw);
        uint32_t copyRows = calcRowCount(dst->desc().Format, sh);

        for (uint32_t r = 0; r < copyRows; r++) {
            const uint8_t *sp = src->cpuData().data() + size_t(sy / bh + r) * srcPitch + size_t(sx / bw) * bb;
            uint8_t *dp = dst->cpuData().data() + size_t(dy / bh + r) * dstPitch + size_t(dx / bw) * bb;
            std::memcpy(dp, sp, copyPitch);
        }
        D3DMtlRect dr = { int32_t(dx), int32_t(dy), int32_t(dx + sw), int32_t(dy + sh) };
        dst->uploadCpuToGpu(&dr);
        return D3D_S_OK;
    }

    if (src->texture() && dst->texture()) {
        if (src->hasAnyPendingClear()) flushSurfacePendingClear(src);
        if (dst->hasAnyPendingClear()) {
            if (!dst->formatInfo().isDepth && dx == 0 && dy == 0 &&
                sw == dst->desc().Width && sh == dst->desc().Height) {
                dst->clearPendingColorClear();
                if (!dst->hasAnyPendingClear()) unregisterPendingClearSurface(dst);
            } else {
                flushSurfacePendingClear(dst);
            }
        }
        endRenderPass();
        flushPendingBlits();
        ensureCommandBuffer();
        id<MTLBlitCommandEncoder> blit = [cmdBuf_ blitCommandEncoder];
        [blit copyFromTexture:src->texture() sourceSlice:src->slice() sourceLevel:src->mipLevel()
                 sourceOrigin:MTLOriginMake(sx, sy, 0) sourceSize:MTLSizeMake(sw, sh, 1)
                    toTexture:dst->texture() destinationSlice:dst->slice() destinationLevel:dst->mipLevel()
            destinationOrigin:MTLOriginMake(dx, dy, 0)];
        [blit endEncoding];
        hasPendingWork_ = true;
        dst->setHasGpuData(true);
        dst->setCpuDataBacked(false);
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::UpdateTexture(IMtlDirect3DBaseTexture9 *pSourceTexture, IMtlDirect3DBaseTexture9 *pDestinationTexture) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pSourceTexture || !pDestinationTexture) return D3D_ERR_INVALIDCALL;
    uint32_t type = pSourceTexture->GetType();
    if (type != pDestinationTexture->GetType()) return D3D_ERR_INVALIDCALL;

    auto matchMipOffsets = [](uint32_t sw, uint32_t sh, uint32_t sCount,
                              uint32_t dw, uint32_t dh, uint32_t dCount,
                              uint32_t &sOff, uint32_t &dOff) -> bool {
        sOff = 0; dOff = 0;
        if (sw == dw && sh == dh) return true;
        if (sw >= dw && sh >= dh) {
            uint32_t w = sw, h = sh;
            for (uint32_t l = 0; l < sCount; l++) {
                if (w == dw && h == dh) { sOff = l; return true; }
                w = std::max(w >> 1, 1u);
                h = std::max(h >> 1, 1u);
            }
        } else if (dw >= sw && dh >= sh) {
            uint32_t w = dw, h = dh;
            for (uint32_t l = 0; l < dCount; l++) {
                if (w == sw && h == sh) { dOff = l; return true; }
                w = std::max(w >> 1, 1u);
                h = std::max(h >> 1, 1u);
            }
        }
        return false;
    };

    if (type == D3DM_RTYPE_TEXTURE) {
        auto *src = static_cast<MtlTexture9 *>(pSourceTexture);
        auto *dst = static_cast<MtlTexture9 *>(pDestinationTexture);
        uint32_t sOff = 0, dOff = 0;
        if (!matchMipOffsets(src->common().width, src->common().height, src->GetLevelCount(),
                             dst->common().width, dst->common().height, dst->GetLevelCount(), sOff, dOff)) {
            return D3D_ERR_INVALIDCALL;
        }
        uint32_t levels = std::min(src->GetLevelCount() - sOff, dst->GetLevelCount() - dOff);
        for (uint32_t l = 0; l < levels; l++) {
            UpdateSurface(src->surfaceAt(sOff + l), nullptr, dst->surfaceAt(dOff + l), nullptr);
        }
    } else if (type == D3DM_RTYPE_CUBETEXTURE) {
        auto *src = static_cast<MtlCubeTexture9 *>(pSourceTexture);
        auto *dst = static_cast<MtlCubeTexture9 *>(pDestinationTexture);
        uint32_t sOff = 0, dOff = 0;
        if (!matchMipOffsets(src->common().width, src->common().height, src->GetLevelCount(),
                             dst->common().width, dst->common().height, dst->GetLevelCount(), sOff, dOff)) {
            return D3D_ERR_INVALIDCALL;
        }
        uint32_t levels = std::min(src->GetLevelCount() - sOff, dst->GetLevelCount() - dOff);
        for (uint32_t f = 0; f < 6; f++) {
            for (uint32_t l = 0; l < levels; l++) {
                UpdateSurface(src->surfaceAt(f, sOff + l), nullptr, dst->surfaceAt(f, dOff + l), nullptr);
            }
        }
    } else if (type == D3DM_RTYPE_VOLUMETEXTURE) {
        auto *src = static_cast<MtlVolumeTexture9 *>(pSourceTexture);
        auto *dst = static_cast<MtlVolumeTexture9 *>(pDestinationTexture);
        uint32_t levels = std::min(src->GetLevelCount(), dst->GetLevelCount());
        for (uint32_t l = 0; l < levels; l++) {
            MtlVolume9 *sv = src->volumeAt(l);
            MtlVolume9 *dv = dst->volumeAt(l);
            if (sv && dv && !sv->cpuData().empty()) {
                dv->ensureCpuBacking();
                size_t n = std::min(sv->cpuData().size(), dv->cpuData().size());
                std::memcpy(dv->cpuData().data(), sv->cpuData().data(), n);
                dv->uploadCpuToGpu();
            }
        }
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetRenderTargetData(IMtlDirect3DSurface9 *pRenderTarget, IMtlDirect3DSurface9 *pDestSurface) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pRenderTarget || !pDestSurface) return D3D_ERR_INVALIDCALL;
    auto *src = static_cast<MtlSurface9 *>(pRenderTarget);
    auto *dst = static_cast<MtlSurface9 *>(pDestSurface);
    if (!src->texture()) return D3D_ERR_INVALIDCALL;
    if (src->hasAnyPendingClear()) flushSurfacePendingClear(src);
    dst->ensureCpuBacking();
    readbackTextureToCpu(src->texture(), src->mipLevel(), src->slice(),
                         src->desc().Format, src->desc().Width, src->desc().Height, dst->cpuData());
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetFrontBufferData(uint32_t iSwapChain, IMtlDirect3DSurface9 *pDestSurface) {
    if (iSwapChain != 0 || !primarySwapChain_) return D3D_ERR_INVALIDCALL;
    return primarySwapChain_->GetFrontBufferData(pDestSurface);
}

struct BlitParamsHost {
    float srcRect[4];
    float dstRect[4];
};

struct ClearParamsHost {
    float dstRect[4];
    float color[4];
    float depth;
    float pad[3];
};

void MtlDirect3DDevice9::blitQuad(id<MTLTexture> srcTex, const D3DMtlRect *srcRect, uint32_t srcW, uint32_t srcH,
                                  id<MTLTexture> dstTex, uint32_t dstLevel, uint32_t dstSlice,
                                  MTLPixelFormat dstFmt, const D3DMtlRect *dstRect, uint32_t dstW, uint32_t dstH,
                                  bool linearFilter) {
    if (!srcTex || !dstTex || srcW == 0 || srcH == 0 || dstW == 0 || dstH == 0) return;

    id<MTLRenderPipelineState> pso = pipelineCache_->getBlitPipeline(dstFmt);
    if (!pso) return;

    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture = dstTex;
    rpd.colorAttachments[0].level   = dstLevel;
    rpd.colorAttachments[0].slice   = dstSlice;
    rpd.colorAttachments[0].loadAction  = (dstRect == nullptr) ? MTLLoadActionDontCare : MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> enc = [cmdBuf_ renderCommandEncoderWithDescriptor:rpd];
    if (!enc) return;

    [enc setRenderPipelineState:pso];
    [enc setViewport:{ 0.0, 0.0, double(dstW), double(dstH), 0.0, 1.0 }];

    float u0 = 0.0f, v0 = 0.0f, u1 = 1.0f, v1 = 1.0f;
    if (srcRect) {
        u0 = float(srcRect->left)   / float(srcW);
        v0 = float(srcRect->top)    / float(srcH);
        u1 = float(srcRect->right)  / float(srcW);
        v1 = float(srcRect->bottom) / float(srcH);
    }
    float x0 = -1.0f, y0 = 1.0f, x1 = 1.0f, y1 = -1.0f;
    if (dstRect) {
        x0 = (float(dstRect->left)   / float(dstW)) * 2.0f - 1.0f;
        x1 = (float(dstRect->right)  / float(dstW)) * 2.0f - 1.0f;
        y0 = 1.0f - (float(dstRect->top)    / float(dstH)) * 2.0f;
        y1 = 1.0f - (float(dstRect->bottom) / float(dstH)) * 2.0f;
    }

    BlitParamsHost bp = {
        { u0, v0, u1 - u0, v1 - v0 },
        { x0, y0, x1 - x0, y1 - y0 }
    };
    [enc setVertexBytes:&bp length:sizeof(bp) atIndex:0];
    [enc setFragmentTexture:srcTex atIndex:0];
    [enc setFragmentSamplerState:(linearFilter ? pipelineCache_->linearClampSampler()
                                               : pipelineCache_->pointClampSampler()) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [enc endEncoding];
    hasPendingWork_ = true;
}

MTL_HRESULT MtlDirect3DDevice9::StretchRect(IMtlDirect3DSurface9 *pSourceSurface, const D3DMtlRect *pSourceRect,
                                            IMtlDirect3DSurface9 *pDestSurface, const D3DMtlRect *pDestRect, uint32_t Filter) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pSourceSurface || !pDestSurface) return D3D_ERR_INVALIDCALL;
    auto *src = static_cast<MtlSurface9 *>(pSourceSurface);
    auto *dst = static_cast<MtlSurface9 *>(pDestSurface);
    if (!src->texture() || !dst->texture()) return D3D_S_OK;

    uint32_t sx = 0, sy = 0, sw = 0, sh = 0;
    if (!validateSurfaceRect(pSourceRect, src->desc().Width, src->desc().Height, sx, sy, sw, sh)) {
        return D3D_ERR_INVALIDCALL;
    }
    uint32_t dx = 0, dy = 0, dw = 0, dh = 0;
    if (!validateSurfaceRect(pDestRect, dst->desc().Width, dst->desc().Height, dx, dy, dw, dh)) {
        return D3D_ERR_INVALIDCALL;
    }
    if (sw == 0 || sh == 0 || dw == 0 || dh == 0) return D3D_S_OK;

    if (src->hasAnyPendingClear()) flushSurfacePendingClear(src);
    if (dst->hasAnyPendingClear()) {
        if (!dst->formatInfo().isDepth && dx == 0 && dy == 0 &&
            dw == dst->desc().Width && dh == dst->desc().Height) {
            dst->clearPendingColorClear();
            if (!dst->hasAnyPendingClear()) unregisterPendingClearSurface(dst);
        } else {
            flushSurfacePendingClear(dst);
        }
    }
    endRenderPass();
    flushPendingBlits();
    ensureCommandBuffer();

    if (src->formatInfo().pixelFormat == dst->formatInfo().pixelFormat && sw == dw && sh == dh) {
        id<MTLBlitCommandEncoder> blit = [cmdBuf_ blitCommandEncoder];
        [blit copyFromTexture:src->texture() sourceSlice:src->slice() sourceLevel:src->mipLevel()
                 sourceOrigin:MTLOriginMake(sx, sy, 0) sourceSize:MTLSizeMake(sw, sh, 1)
                    toTexture:dst->texture() destinationSlice:dst->slice() destinationLevel:dst->mipLevel()
            destinationOrigin:MTLOriginMake(dx, dy, 0)];
        [blit endEncoding];
        hasPendingWork_ = true;
        dst->setHasGpuData(true);
        dst->setCpuDataBacked(false);
        return D3D_S_OK;
    }

    if (dst->formatInfo().isDepth) return D3D_S_OK;

    id<MTLTexture> srcView = src->texture();
    if (src->mipLevel() > 0 || src->slice() > 0) {
        srcView = [src->texture() newTextureViewWithPixelFormat:src->formatInfo().pixelFormat
                                                    textureType:MTLTextureType2D
                                                         levels:NSMakeRange(src->mipLevel(), 1)
                                                         slices:NSMakeRange(src->slice(), 1)];
    }

    blitQuad(srcView, pSourceRect, src->desc().Width, src->desc().Height,
             dst->texture(), dst->mipLevel(), dst->slice(), dst->formatInfo().pixelFormat,
             pDestRect, dst->desc().Width, dst->desc().Height,
             Filter == D3DM_TEXF_LINEAR);
    dst->setHasGpuData(true);
    dst->setCpuDataBacked(false);
    return D3D_S_OK;
}

void MtlDirect3DDevice9::clearRectQuad(MtlSurface9 *rt, MtlSurface9 *ds, const D3DMtlRect &rect,
                                       uint32_t flags, D3DCOLOR_T color, float z, uint32_t stencil) {
    if (rect.right <= rect.left || rect.bottom <= rect.top) return;

    bool doColor   = (flags & D3DM_CLEAR_TARGET) && rt && !rt->isNull() && rt->texture();
    bool doDepth   = (flags & D3DM_CLEAR_ZBUFFER) && ds && ds->texture();
    bool doStencil = (flags & D3DM_CLEAR_STENCIL) && ds && ds->texture() && ds->formatInfo().hasStencil;
    if (!doColor && !doDepth && !doStencil) return;

    uint32_t w = rt ? rt->desc().Width : ds->desc().Width;
    uint32_t h = rt ? rt->desc().Height : ds->desc().Height;

    MTLPixelFormat colorFmt = doColor ? rt->formatInfo().pixelFormat : MTLPixelFormatInvalid;
    MTLPixelFormat dsFmt = (doDepth || doStencil) ? ds->formatInfo().pixelFormat : MTLPixelFormatInvalid;
    id<MTLRenderPipelineState> pso = pipelineCache_->getClearPipeline(colorFmt, dsFmt);
    if (!pso) return;

    ensureCommandBuffer();
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.renderTargetWidth = w;
    rpd.renderTargetHeight = h;
    if (doColor) {
        rpd.colorAttachments[0].texture     = rt->texture();
        rpd.colorAttachments[0].level       = rt->mipLevel();
        rpd.colorAttachments[0].slice       = rt->slice();
        rpd.colorAttachments[0].loadAction  = MTLLoadActionLoad;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    }
    if (doDepth || doStencil) {
        rpd.depthAttachment.texture     = ds->texture();
        rpd.depthAttachment.level       = ds->mipLevel();
        rpd.depthAttachment.slice       = ds->slice();
        rpd.depthAttachment.loadAction  = MTLLoadActionLoad;
        rpd.depthAttachment.storeAction = MTLStoreActionStore;
        if (ds->formatInfo().hasStencil) {
            rpd.stencilAttachment.texture     = ds->texture();
            rpd.stencilAttachment.level       = ds->mipLevel();
            rpd.stencilAttachment.slice       = ds->slice();
            rpd.stencilAttachment.loadAction  = MTLLoadActionLoad;
            rpd.stencilAttachment.storeAction = MTLStoreActionStore;
        }
    }

    MtlDepthStencilKey dsk = {};
    dsk.hasDepth      = doDepth ? 1 : 0;
    dsk.zEnable       = doDepth ? 1 : 0;
    dsk.zWrite        = doDepth ? 1 : 0;
    dsk.zFunc         = D3DM_CMP_ALWAYS;
    dsk.hasStencil    = doStencil ? 1 : 0;
    dsk.stencilEnable = doStencil ? 1 : 0;
    dsk.frontFunc     = D3DM_CMP_ALWAYS;
    dsk.frontFail     = D3DM_STENCILOP_REPLACE;
    dsk.frontZFail    = D3DM_STENCILOP_REPLACE;
    dsk.frontPass     = D3DM_STENCILOP_REPLACE;
    dsk.readMask      = 0xFF;
    dsk.writeMask     = 0xFF;
    id<MTLDepthStencilState> dss = pipelineCache_->getOrCreateDepthStencilState(dsk);

    id<MTLRenderCommandEncoder> enc = [cmdBuf_ renderCommandEncoderWithDescriptor:rpd];
    if (!enc) return;
    [enc setRenderPipelineState:pso];
    [enc setDepthStencilState:dss];
    [enc setStencilReferenceValue:(stencil & 0xFFu)];
    [enc setViewport:{ 0.0, 0.0, double(w), double(h), 0.0, 1.0 }];

    float x0 = (float(rect.left)   / float(w)) * 2.0f - 1.0f;
    float x1 = (float(rect.right)  / float(w)) * 2.0f - 1.0f;
    float y0 = 1.0f - (float(rect.top)    / float(h)) * 2.0f;
    float y1 = 1.0f - (float(rect.bottom) / float(h)) * 2.0f;

    ClearParamsHost cp = {};
    cp.dstRect[0] = x0; cp.dstRect[1] = y0; cp.dstRect[2] = x1 - x0; cp.dstRect[3] = y1 - y0;
    cp.color[0] = float((color >> 16) & 0xFFu) / 255.0f;
    cp.color[1] = float((color >>  8) & 0xFFu) / 255.0f;
    cp.color[2] = float((color >>  0) & 0xFFu) / 255.0f;
    cp.color[3] = float((color >> 24) & 0xFFu) / 255.0f;
    cp.depth = z;

    [enc setVertexBytes:&cp length:sizeof(cp) atIndex:0];
    if (doColor) [enc setFragmentBytes:&cp length:sizeof(cp) atIndex:0];
    [enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [enc endEncoding];
    hasPendingWork_ = true;
}

MTL_HRESULT MtlDirect3DDevice9::ColorFill(IMtlDirect3DSurface9 *pSurface, const D3DMtlRect *pRect, D3DCOLOR_T color) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pSurface) return D3D_ERR_INVALIDCALL;
    auto *surf = static_cast<MtlSurface9 *>(pSurface);
    if (surf->isNull() || !surf->texture()) return D3D_S_OK;

    bool fullRect = (!pRect) ||
        (pRect->left <= 0 && pRect->top <= 0 &&
         pRect->right >= int32_t(surf->desc().Width) && pRect->bottom >= int32_t(surf->desc().Height));

    if (fullRect) {
        if (encoder_ != nil && passHasColorAttach_) {
            uint8_t discardC = 0;
            for (int i = 0; i < 4; i++) {
                if ((passColorMask_ & uint8_t(1u << i)) && passColorSurfaces_[i] == surf) {
                    discardC |= uint8_t(1u << i);
                }
            }
            if (discardC != 0) {
                endRenderPass(false, false, discardC);
            }
        }
        flushPendingBlits();
        surf->setPendingColorClear(color);
        registerPendingClearSurface(surf);
    } else {
        if (surf->hasAnyPendingClear()) flushSurfacePendingClear(surf);
        endRenderPass();
        flushPendingBlits();
        ensureCommandBuffer();
        clearRectQuad(surf, nullptr, *pRect, D3DM_CLEAR_TARGET, color, 1.0f, 0);
    }
    return D3D_S_OK;
}

/* ========================================================================= */
/* Render Targets, Depth-Stencil, Scene, Clear                               */
/* ========================================================================= */

MTL_HRESULT MtlDirect3DDevice9::SetRenderTarget(uint32_t RenderTargetIndex, IMtlDirect3DSurface9 *pRenderTarget) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (RenderTargetIndex >= 4) return D3D_ERR_INVALIDCALL;
    auto *newSurf = static_cast<MtlSurface9 *>(pRenderTarget);
    if (RenderTargetIndex == 0 && !newSurf) return D3D_ERR_INVALIDCALL;

    if (renderTargets_[RenderTargetIndex] != newSurf) {
        endRenderPass();

        if (RenderTargetIndex == 0) {
            maybeSplitAfterOcclusionBatch(8u);
        }

        if (newSurf) newSurf->AddRef();
        if (renderTargets_[RenderTargetIndex]) renderTargets_[RenderTargetIndex]->Release();
        renderTargets_[RenderTargetIndex] = newSurf;
    }

    /* D3D9 contract: SetRenderTarget(0, ...) resets the viewport to the full size of RT0 */
    if (RenderTargetIndex == 0 && newSurf) {
        state_.viewport = { 0, 0, newSurf->desc().Width, newSurf->desc().Height, 0.0f, 1.0f };
        vpDirty_ = true;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetRenderTarget(uint32_t RenderTargetIndex, IMtlDirect3DSurface9 **ppRenderTarget) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (RenderTargetIndex >= 4 || !ppRenderTarget) return D3D_ERR_INVALIDCALL;
    if (!renderTargets_[RenderTargetIndex]) {
        *ppRenderTarget = nullptr;
        return D3D_ERR_NOTAVAILABLE;
    }
    renderTargets_[RenderTargetIndex]->AddRef();
    *ppRenderTarget = renderTargets_[RenderTargetIndex];
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetDepthStencilSurface(IMtlDirect3DSurface9 *pNewZStencil) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    auto *newDs = static_cast<MtlSurface9 *>(pNewZStencil);
    if (depthStencil_ != newDs) {
        endRenderPass();
        maybeSplitAfterOcclusionBatch(newDs == nullptr ? 1u : 8u);

        if (newDs) newDs->AddRef();
        if (depthStencil_) depthStencil_->Release();
        depthStencil_ = newDs;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetDepthStencilSurface(IMtlDirect3DSurface9 **ppZStencilSurface) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!ppZStencilSurface) return D3D_ERR_INVALIDCALL;
    if (!depthStencil_) {
        *ppZStencilSurface = nullptr;
        return D3D_ERR_NOTAVAILABLE;
    }
    depthStencil_->AddRef();
    *ppZStencilSurface = depthStencil_;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::BeginScene() {
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::EndScene() {
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::Clear(uint32_t Count, const D3DMtlRect *pRects, uint32_t Flags,
                                      D3DCOLOR_T Color, float Z, uint32_t Stencil) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (Flags == 0) return D3D_S_OK;

    uint32_t rtW = 0, rtH = 0;
    if (renderTargets_[0]) {
        rtW = renderTargets_[0]->desc().Width;
        rtH = renderTargets_[0]->desc().Height;
    } else if (depthStencil_) {
        rtW = depthStencil_->desc().Width;
        rtH = depthStencil_->desc().Height;
    }
    if (rtW == 0 || rtH == 0) return D3D_S_OK;

    /* Determine effective clip region (viewport + optional scissor) */
    int32_t clipL = int32_t(state_.viewport.X);
    int32_t clipT = int32_t(state_.viewport.Y);
    int32_t clipR = int32_t(state_.viewport.X + state_.viewport.Width);
    int32_t clipB = int32_t(state_.viewport.Y + state_.viewport.Height);
    clipL = std::clamp(clipL, 0, int32_t(rtW));
    clipT = std::clamp(clipT, 0, int32_t(rtH));
    clipR = std::clamp(clipR, clipL, int32_t(rtW));
    clipB = std::clamp(clipB, clipT, int32_t(rtH));

    if (state_.renderStates[D3DM_RS_SCISSORTESTENABLE]) {
        clipL = std::max(clipL, state_.scissorRect.left);
        clipT = std::max(clipT, state_.scissorRect.top);
        clipR = std::min(clipR, state_.scissorRect.right);
        clipB = std::min(clipB, state_.scissorRect.bottom);
    }
    if (clipR <= clipL || clipB <= clipT) return D3D_S_OK;

    bool fullClip = (clipL == 0 && clipT == 0 && clipR == int32_t(rtW) && clipB == int32_t(rtH));
    bool fullClear = fullClip && (Count == 0 || !pRects ||
        (Count == 1 && pRects[0].left <= 0 && pRects[0].top <= 0 &&
         pRects[0].right >= int32_t(rtW) && pRects[0].bottom >= int32_t(rtH)));

    if (fullClear) {
        if (encoder_ != nil) {
            bool sameDs = (passDsSurface_ != nullptr && passDsSurface_ == depthStencil_);
            bool discardD = sameDs && ((Flags & D3DM_CLEAR_ZBUFFER) != 0);
            bool discardS = sameDs && ((Flags & D3DM_CLEAR_STENCIL) != 0);
            uint8_t discardC = 0;
            if ((Flags & D3DM_CLEAR_TARGET) != 0 && passHasColorAttach_) {
                for (int i = 0; i < 4; i++) {
                    if ((passColorMask_ & uint8_t(1u << i)) &&
                        renderTargets_[i] && passColorSurfaces_[i] == renderTargets_[i]) {
                        discardC |= uint8_t(1u << i);
                    }
                }
            }
            endRenderPass(discardD, discardS, discardC);
        }
        if (Flags & (D3DM_CLEAR_TARGET | D3DM_CLEAR_ZBUFFER)) {
            maybeSplitAfterOcclusionBatch(8u);
        }
        if (Flags & D3DM_CLEAR_TARGET) {
            for (int i = 0; i < 4; i++) {
                MtlSurface9 *rt = renderTargets_[i];
                if (!rt || rt->isNull() || !rt->texture()) continue;
                rt->setPendingColorClear(Color);
                registerPendingClearSurface(rt);
            }
        }
        if (depthStencil_ && depthStencil_->texture()) {
            if (Flags & D3DM_CLEAR_ZBUFFER) {
                depthStencil_->setPendingZClear(Z);
                registerPendingClearSurface(depthStencil_);
            }
            if ((Flags & D3DM_CLEAR_STENCIL) && depthStencil_->formatInfo().hasStencil) {
                depthStencil_->setPendingSClear(Stencil);
                registerPendingClearSurface(depthStencil_);
            }
        }
        return D3D_S_OK;
    }

    /* Partial / sub-rect clear: flush any prior pending clear on target surfaces, close open pass, clear quads */
    if ((Flags & D3DM_CLEAR_TARGET) && renderTargets_[0] && renderTargets_[0]->hasAnyPendingClear()) {
        flushSurfacePendingClear(renderTargets_[0]);
    }
    if ((Flags & (D3DM_CLEAR_ZBUFFER | D3DM_CLEAR_STENCIL)) && depthStencil_ && depthStencil_->hasAnyPendingClear()) {
        flushSurfacePendingClear(depthStencil_);
    }
    endRenderPass();
    flushPendingBlits();

    auto emitRect = [&](const D3DMtlRect &r) {
        D3DMtlRect c = {
            std::max(r.left, clipL), std::max(r.top, clipT),
            std::min(r.right, clipR), std::min(r.bottom, clipB)
        };
        if (c.right > c.left && c.bottom > c.top) {
            clearRectQuad(renderTargets_[0], depthStencil_, c, Flags, Color, Z, Stencil);
        }
    };

    if (Count > 0 && pRects) {
        for (uint32_t i = 0; i < Count; i++) emitRect(pRects[i]);
    } else {
        emitRect({ clipL, clipT, clipR, clipB });
    }
    return D3D_S_OK;
}

/* ========================================================================= */
/* State Getters / Setters                                                   */
/* ========================================================================= */

MTL_HRESULT MtlDirect3DDevice9::SetTransform(uint32_t State, const float *pMatrix) {
    if (!pMatrix) return D3D_ERR_INVALIDCALL;
    if (State == 2) std::memcpy(state_.transforms[0], pMatrix, 64);
    else if (State == 3) std::memcpy(state_.transforms[1], pMatrix, 64);
    else if (State == 256) std::memcpy(state_.transforms[2], pMatrix, 64);
    else if (State == 257) std::memcpy(state_.transforms[3], pMatrix, 64);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetTransform(uint32_t State, float *pMatrix) {
    if (!pMatrix) return D3D_ERR_INVALIDCALL;
    if (State == 2) std::memcpy(pMatrix, state_.transforms[0], 64);
    else if (State == 3) std::memcpy(pMatrix, state_.transforms[1], 64);
    else if (State == 256) std::memcpy(pMatrix, state_.transforms[2], 64);
    else if (State == 257) std::memcpy(pMatrix, state_.transforms[3], 64);
    else {
        std::memset(pMatrix, 0, 64);
        pMatrix[0] = pMatrix[5] = pMatrix[10] = pMatrix[15] = 1.0f;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::MultiplyTransform(uint32_t State, const float *pMatrix) {
    return SetTransform(State, pMatrix);
}

MTL_HRESULT MtlDirect3DDevice9::SetViewport(const D3DMtlViewport *pViewport) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pViewport) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordViewport(*pViewport); return D3D_S_OK; }
    state_.viewport = *pViewport;
    vpDirty_ = true;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetViewport(D3DMtlViewport *pViewport) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pViewport) return D3D_ERR_INVALIDCALL;
    *pViewport = state_.viewport;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetClipPlane(uint32_t Index, const float *pPlane) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (Index >= 6 || !pPlane) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordClipPlane(Index, pPlane); return D3D_S_OK; }
    std::memcpy(state_.clipPlanes[Index], pPlane, 16);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetClipPlane(uint32_t Index, float *pPlane) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (Index >= 6 || !pPlane) return D3D_ERR_INVALIDCALL;
    std::memcpy(pPlane, state_.clipPlanes[Index], 16);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetRenderState(uint32_t State, uint32_t Value) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (State >= D3DM_RS_MAX) return D3D_S_OK;
    if (recordingBlock_) { recordingBlock_->recordRenderState(State, Value); return D3D_S_OK; }
    if (state_.renderStates[State] == Value) return D3D_S_OK;
    uint32_t prevVal = state_.renderStates[State];
    state_.renderStates[State] = Value;
    switch (State) {
        case D3DM_RS_COLORWRITEENABLE:
            psoDirty_ = true;
            if (prevVal == 0u && Value != 0u) {
                maybeSplitAfterOcclusionBatch();
            }
            break;
        case D3DM_RS_ALPHABLENDENABLE:
        case D3DM_RS_SRCBLEND:
        case D3DM_RS_DESTBLEND:
        case D3DM_RS_BLENDOP:
        case D3DM_RS_SEPARATEALPHABLENDENABLE:
        case D3DM_RS_SRCBLENDALPHA:
        case D3DM_RS_DESTBLENDALPHA:
        case D3DM_RS_BLENDOPALPHA:
        case D3DM_RS_COLORWRITEENABLE1:
        case D3DM_RS_COLORWRITEENABLE2:
        case D3DM_RS_COLORWRITEENABLE3:
        case D3DM_RS_ADAPTIVETESS_Y:
        case D3DM_RS_SHADEMODE:
        case D3DM_RS_ALPHATESTENABLE:
            psoDirty_ = true;
            break;
        case D3DM_RS_ZENABLE:
        case D3DM_RS_ZWRITEENABLE:
        case D3DM_RS_ZFUNC:
        case D3DM_RS_STENCILENABLE:
        case D3DM_RS_STENCILFAIL:
        case D3DM_RS_STENCILZFAIL:
        case D3DM_RS_STENCILPASS:
        case D3DM_RS_STENCILFUNC:
        case D3DM_RS_STENCILMASK:
        case D3DM_RS_STENCILWRITEMASK:
        case D3DM_RS_TWOSIDEDSTENCILMODE:
        case D3DM_RS_CCW_STENCILFAIL:
        case D3DM_RS_CCW_STENCILZFAIL:
        case D3DM_RS_CCW_STENCILPASS:
        case D3DM_RS_CCW_STENCILFUNC:
            dssDirty_ = true;
            break;
        case D3DM_RS_CULLMODE:            cullDirty_ = true; break;
        case D3DM_RS_FILLMODE:            fillDirty_ = true; break;
        case D3DM_RS_SLOPESCALEDEPTHBIAS: slopeDirty_ = true; break;
        case D3DM_RS_BLENDFACTOR:         blendColorDirty_ = true; break;
        case D3DM_RS_STENCILREF:          stencilRefDirty_ = true; break;
        case D3DM_RS_SCISSORTESTENABLE:   scissorDirty_ = true; break;
        case D3DM_RS_FOGENABLE:
        case D3DM_RS_FOGCOLOR:
        case D3DM_RS_FOGTABLEMODE:
        case D3DM_RS_FOGSTART:
        case D3DM_RS_FOGEND:
        case D3DM_RS_FOGDENSITY:
            vsFogDirty_ = psFogDirty_ = true;
            break;
        default: break;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetRenderState(uint32_t State, uint32_t *pValue) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pValue) return D3D_ERR_INVALIDCALL;
    *pValue = (State < D3DM_RS_MAX) ? state_.renderStates[State] : 0u;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateStateBlock(uint32_t Type, IMtlDirect3DStateBlock9 **ppSB) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!ppSB) return D3D_ERR_INVALIDCALL;
    *ppSB = new MtlStateBlock9(this, Type);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::BeginStateBlock() {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (recordingBlock_) return D3D_ERR_INVALIDCALL;
    recordingBlock_ = new MtlStateBlock9(this, 0);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::EndStateBlock(IMtlDirect3DStateBlock9 **ppSB) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!recordingBlock_ || !ppSB) return D3D_ERR_INVALIDCALL;
    *ppSB = recordingBlock_;
    recordingBlock_ = nullptr;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetTexture(uint32_t Stage, IMtlDirect3DBaseTexture9 **ppTexture) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!ppTexture) return D3D_ERR_INVALIDCALL;
    int slot = mapD3dSamplerSlot(Stage);
    if (slot < 0) { *ppTexture = nullptr; return D3D_S_OK; }
    if (state_.textures[slot]) state_.textures[slot]->AddRef();
    *ppTexture = state_.textures[slot];
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetTexture(uint32_t Stage, IMtlDirect3DBaseTexture9 *pTexture) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    int slot = mapD3dSamplerSlot(Stage);
    if (slot < 0) return D3D_S_OK;
    if (recordingBlock_) { recordingBlock_->recordTexture(uint32_t(slot), pTexture); return D3D_S_OK; }
    if (state_.textures[slot] == pTexture) return D3D_S_OK;
    if (pTexture) pTexture->AddRef();
    if (state_.textures[slot]) state_.textures[slot]->Release();
    state_.textures[slot] = pTexture;
    if (slot < 16) psTexDirtyMask_ |= uint16_t(1u << slot);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetTextureStageState(uint32_t Stage, uint32_t Type, uint32_t *pValue) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pValue) return D3D_ERR_INVALIDCALL;
    *pValue = (Stage < kMaxTexStages && Type < kMaxTexStageStates) ? state_.texStageStates[Stage][Type] : 0u;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetTextureStageState(uint32_t Stage, uint32_t Type, uint32_t Value) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (Stage >= kMaxTexStages || Type >= kMaxTexStageStates) return D3D_S_OK;
    if (recordingBlock_) { recordingBlock_->recordTexStageState(Stage, Type, Value); return D3D_S_OK; }
    if (state_.texStageStates[Stage][Type] == Value) return D3D_S_OK;
    state_.texStageStates[Stage][Type] = Value;
    psTexStageDirty_ = true;
    if (Type == 24) psSampDirtyMask_ |= uint16_t(1u << Stage);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetSamplerState(uint32_t Sampler, uint32_t Type, uint32_t *pValue) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pValue) return D3D_ERR_INVALIDCALL;
    int slot = mapD3dSamplerSlot(Sampler);
    *pValue = (slot >= 0 && Type < D3DM_SAMP_MAX) ? state_.samplerStates[slot][Type] : 0u;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetSamplerState(uint32_t Sampler, uint32_t Type, uint32_t Value) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    int slot = mapD3dSamplerSlot(Sampler);
    if (slot < 0 || Type >= D3DM_SAMP_MAX) return D3D_S_OK;
    if (recordingBlock_) { recordingBlock_->recordSamplerState(uint32_t(slot), Type, Value); return D3D_S_OK; }
    if (Type == D3DM_SAMP_MIPMAPLODBIAS) {
        if (Value == D3D_MAKEFOURCC('G', 'E', 'T', '4')) {
            if (!(state_.fetch4Mask & (1u << slot))) {
                state_.fetch4Mask |= (1u << slot);
                if (slot < 16) psSampDirtyMask_ |= uint16_t(1u << slot);
            }
            return D3D_S_OK;
        }
        if (Value == D3D_MAKEFOURCC('G', 'E', 'T', '1')) {
            if (state_.fetch4Mask & (1u << slot)) {
                state_.fetch4Mask &= ~(1u << slot);
                if (slot < 16) psSampDirtyMask_ |= uint16_t(1u << slot);
            }
            return D3D_S_OK;
        }
    }
    if (state_.samplerStates[slot][Type] == Value) return D3D_S_OK;
    state_.samplerStates[slot][Type] = Value;
    if (slot < 16) {
        if (Type == D3DM_SAMP_SRGBTEXTURE) psTexDirtyMask_ |= uint16_t(1u << slot);
        else psSampDirtyMask_ |= uint16_t(1u << slot);
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetScissorRect(const D3DMtlRect *pRect) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pRect) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordScissor(*pRect); return D3D_S_OK; }
    state_.scissorRect = *pRect;
    scissorDirty_ = true;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetScissorRect(D3DMtlRect *pRect) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pRect) return D3D_ERR_INVALIDCALL;
    *pRect = state_.scissorRect;
    return D3D_S_OK;
}

/* ========================================================================= */
/* Vertex Declarations, Shaders, Streams, Constants, Queries                 */
/* ========================================================================= */

MTL_HRESULT MtlDirect3DDevice9::CreateVertexDeclaration(const D3DMtlVertexElement *pVertexElements, IMtlDirect3DVertexDeclaration9 **ppDecl) {
    if (!pVertexElements || !ppDecl) return D3D_ERR_INVALIDCALL;
    *ppDecl = new MtlVertexDeclaration9(this, pVertexElements);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetVertexDeclaration(IMtlDirect3DVertexDeclaration9 *pDecl) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    auto *d = static_cast<MtlVertexDeclaration9 *>(pDecl);
    if (recordingBlock_) { recordingBlock_->recordVdecl(d); return D3D_S_OK; }
    if (state_.vdecl != d) {
        if (d) d->AddRef();
        if (state_.vdecl) state_.vdecl->Release();
        state_.vdecl = d;
        psoDirty_ = true;
    }
    state_.fvf = 0;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetVertexDeclaration(IMtlDirect3DVertexDeclaration9 **ppDecl) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!ppDecl) return D3D_ERR_INVALIDCALL;
    if (state_.vdecl) state_.vdecl->AddRef();
    *ppDecl = state_.vdecl;
    return D3D_S_OK;
}

MtlVertexDeclaration9 *MtlDirect3DDevice9::getDeclForFvf(uint32_t fvf) {
    auto it = fvfDeclCache_.find(fvf);
    if (it != fvfDeclCache_.end()) return it->second;

    std::vector<D3DMtlVertexElement> elems;
    uint16_t off = 0;
    uint32_t posMask = fvf & 0x0Eu;
    if (posMask == 0x02u) { /* XYZ */
        elems.push_back({0, off, D3DM_DECLTYPE_FLOAT3, 0, 0, 0}); off += 12;
    } else if (posMask == 0x04u) { /* XYZRHW */
        elems.push_back({0, off, D3DM_DECLTYPE_FLOAT4, 0, 9, 0}); off += 16;
    } else if (posMask >= 0x06u) {
        elems.push_back({0, off, D3DM_DECLTYPE_FLOAT3, 0, 0, 0}); off += 12;
        uint32_t betas = ((posMask - 0x06u) >> 1) + 1u;
        if (fvf & 0x1000u) betas--;
        if (betas >= 1 && betas <= 4) {
            elems.push_back({0, off, uint8_t(D3DM_DECLTYPE_FLOAT1 + betas - 1), 0, 1, 0});
            off += uint16_t(betas * 4);
        }
        if (fvf & 0x1000u) {
            elems.push_back({0, off, D3DM_DECLTYPE_UBYTE4, 0, 2, 0}); off += 4;
        }
    }
    if (fvf & 0x10u) { elems.push_back({0, off, D3DM_DECLTYPE_FLOAT3, 0, 3, 0}); off += 12; }
    if (fvf & 0x20u) { elems.push_back({0, off, D3DM_DECLTYPE_FLOAT1, 0, 4, 0}); off += 4; }
    if (fvf & 0x40u) { elems.push_back({0, off, D3DM_DECLTYPE_D3DCOLOR, 0, 10, 0}); off += 4; }
    if (fvf & 0x80u) { elems.push_back({0, off, D3DM_DECLTYPE_D3DCOLOR, 0, 10, 1}); off += 4; }
    uint32_t texCount = (fvf >> 8) & 0x0Fu;
    for (uint32_t t = 0; t < texCount && t < 8; t++) {
        uint32_t szCode = (fvf >> (16 + t * 2)) & 0x3u;
        uint8_t dt = D3DM_DECLTYPE_FLOAT2; uint16_t bytes = 8;
        if (szCode == 3)      { dt = D3DM_DECLTYPE_FLOAT1; bytes = 4; }
        else if (szCode == 1) { dt = D3DM_DECLTYPE_FLOAT3; bytes = 12; }
        else if (szCode == 2) { dt = D3DM_DECLTYPE_FLOAT4; bytes = 16; }
        elems.push_back({0, off, dt, 0, 5, uint8_t(t)}); off += bytes;
    }
    elems.push_back({0xFF, 0, D3DM_DECLTYPE_UNUSED, 0, 0, 0});

    auto *decl = new MtlVertexDeclaration9(this, elems.data());
    fvfDeclCache_.emplace(fvf, decl);
    return decl;
}

MTL_HRESULT MtlDirect3DDevice9::SetFVF(uint32_t FVF) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (recordingBlock_) { recordingBlock_->recordFvf(FVF); return D3D_S_OK; }
    state_.fvf = FVF;
    if (FVF != 0) {
        MtlVertexDeclaration9 *d = getDeclForFvf(FVF);
        if (state_.vdecl != d) {
            if (d) d->AddRef();
            if (state_.vdecl) state_.vdecl->Release();
            state_.vdecl = d;
            psoDirty_ = true;
        }
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetFVF(uint32_t *pFVF) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pFVF) return D3D_ERR_INVALIDCALL;
    *pFVF = state_.fvf;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateVertexShader(const uint32_t *pFunction, IMtlDirect3DVertexShader9 **ppShader) {
    if (!pFunction || !ppShader) return D3D_ERR_INVALIDCALL;
    auto t0 = stallLogEnabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto parsed = shaderCompiler_->compileFromBytecode(pFunction, true);
    if (stallLogEnabled_) {
        uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
        stallVsCreateCnt_.fetch_add(1u, std::memory_order_relaxed);
        stallVsCreateNs_.fetch_add(ns, std::memory_order_relaxed);
    }
    if (!parsed) return D3D_ERR_INVALIDCALL;
    *ppShader = new MtlVertexShader9(this, std::move(parsed));
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetVertexShader(IMtlDirect3DVertexShader9 *pShader) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    auto *s = static_cast<MtlVertexShader9 *>(pShader);
    if (recordingBlock_) { recordingBlock_->recordVs(s); return D3D_S_OK; }
    if (state_.vs != s) {
        if (s) s->AddRef();
        if (state_.vs) state_.vs->Release();
        state_.vs = s;
        psoDirty_ = true;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetVertexShader(IMtlDirect3DVertexShader9 **ppShader) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!ppShader) return D3D_ERR_INVALIDCALL;
    if (state_.vs) state_.vs->AddRef();
    *ppShader = state_.vs;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetVertexShaderConstantF(uint32_t StartRegister, const float *pConstantData, uint32_t Vector4fCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordVsConstF(StartRegister, pConstantData, Vector4fCount); return D3D_S_OK; }
    if (StartRegister < 256) {
        uint32_t n = std::min(Vector4fCount, 256u - StartRegister);
        size_t bytes = size_t(n) * 16u;
        if (bytes > 0 && std::memcmp(state_.vsConstF[StartRegister], pConstantData, bytes) != 0) {
            std::memcpy(state_.vsConstF[StartRegister], pConstantData, bytes);
            vsConstFVersion_++;
        }
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetVertexShaderConstantF(uint32_t StartRegister, float *pConstantData, uint32_t Vector4fCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (StartRegister < 256) {
        uint32_t n = std::min(Vector4fCount, 256u - StartRegister);
        std::memcpy(pConstantData, state_.vsConstF[StartRegister], size_t(n) * 16u);
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetVertexShaderConstantI(uint32_t StartRegister, const int32_t *pConstantData, uint32_t Vector4iCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordVsConstI(StartRegister, pConstantData, Vector4iCount); return D3D_S_OK; }
    if (StartRegister < 16) {
        uint32_t n = std::min(Vector4iCount, 16u - StartRegister);
        std::memcpy(state_.vsConstI[StartRegister], pConstantData, size_t(n) * 16u);
        vsConstIDirty_ = true;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetVertexShaderConstantI(uint32_t StartRegister, int32_t *pConstantData, uint32_t Vector4iCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (StartRegister < 16) {
        uint32_t n = std::min(Vector4iCount, 16u - StartRegister);
        std::memcpy(pConstantData, state_.vsConstI[StartRegister], size_t(n) * 16u);
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetVertexShaderConstantB(uint32_t StartRegister, const uint32_t *pConstantData, uint32_t BoolCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordVsConstB(StartRegister, pConstantData, BoolCount); return D3D_S_OK; }
    for (uint32_t i = 0; i < BoolCount && StartRegister + i < 16; i++) {
        state_.vsConstB[StartRegister + i] = pConstantData[i] ? 1u : 0u;
    }
    vsConstBDirty_ = true;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetVertexShaderConstantB(uint32_t StartRegister, uint32_t *pConstantData, uint32_t BoolCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    for (uint32_t i = 0; i < BoolCount && StartRegister + i < 16; i++) {
        pConstantData[i] = state_.vsConstB[StartRegister + i];
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetStreamSource(uint32_t StreamNumber, IMtlDirect3DVertexBuffer9 *pStreamData, uint32_t OffsetInBytes, uint32_t Stride) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (StreamNumber >= 16) return D3D_ERR_INVALIDCALL;
    auto *vb = static_cast<MtlVertexBuffer9 *>(pStreamData);
    if (recordingBlock_) { recordingBlock_->recordStream(StreamNumber, vb, OffsetInBytes, Stride); return D3D_S_OK; }
    bool hadVb = (state_.streams[StreamNumber].vb != nullptr);
    bool hasVb = (vb != nullptr);
    if (hadVb != hasVb || state_.streams[StreamNumber].stride != Stride) {
        psoDirty_ = true;
    }
    if (state_.streams[StreamNumber].vb != vb) {
        if (vb) vb->AddRef();
        if (state_.streams[StreamNumber].vb) state_.streams[StreamNumber].vb->Release();
        state_.streams[StreamNumber].vb = vb;
    }
    state_.streams[StreamNumber].offset = OffsetInBytes;
    state_.streams[StreamNumber].stride = Stride;
    streamDirtyMask_ |= uint16_t(1u << StreamNumber);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetStreamSource(uint32_t StreamNumber, IMtlDirect3DVertexBuffer9 **ppStreamData, uint32_t *pOffsetInBytes, uint32_t *pStride) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (StreamNumber >= 16) return D3D_ERR_INVALIDCALL;
    if (ppStreamData) {
        if (state_.streams[StreamNumber].vb) state_.streams[StreamNumber].vb->AddRef();
        *ppStreamData = state_.streams[StreamNumber].vb;
    }
    if (pOffsetInBytes) *pOffsetInBytes = state_.streams[StreamNumber].offset;
    if (pStride) *pStride = state_.streams[StreamNumber].stride;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetStreamSourceFreq(uint32_t StreamNumber, uint32_t Setting) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (StreamNumber >= 16) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordStreamFreq(StreamNumber, Setting); return D3D_S_OK; }
    if (state_.streams[StreamNumber].freq != Setting) {
        state_.streams[StreamNumber].freq = Setting;
        psoDirty_ = true;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetStreamSourceFreq(uint32_t StreamNumber, uint32_t *pSetting) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (StreamNumber >= 16 || !pSetting) return D3D_ERR_INVALIDCALL;
    *pSetting = state_.streams[StreamNumber].freq;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetIndices(IMtlDirect3DIndexBuffer9 *pIndexData) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    auto *ib = static_cast<MtlIndexBuffer9 *>(pIndexData);
    if (recordingBlock_) { recordingBlock_->recordIndices(ib); return D3D_S_OK; }
    if (state_.ib != ib) {
        if (ib) ib->AddRef();
        if (state_.ib) state_.ib->Release();
        state_.ib = ib;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetIndices(IMtlDirect3DIndexBuffer9 **ppIndexData) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!ppIndexData) return D3D_ERR_INVALIDCALL;
    if (state_.ib) state_.ib->AddRef();
    *ppIndexData = state_.ib;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreatePixelShader(const uint32_t *pFunction, IMtlDirect3DPixelShader9 **ppShader) {
    if (!pFunction || !ppShader) return D3D_ERR_INVALIDCALL;
    auto t0 = stallLogEnabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto handle = shaderCompiler_->createPixelShaderAsync(pFunction);
    if (stallLogEnabled_) {
        uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count());
        stallPsCreateCnt_.fetch_add(1u, std::memory_order_relaxed);
        stallPsCreateNs_.fetch_add(ns, std::memory_order_relaxed);
    }
    if (!handle) return D3D_ERR_INVALIDCALL;
    *ppShader = new MtlPixelShader9(this, shaderCompiler_.get(), std::move(handle));
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetPixelShader(IMtlDirect3DPixelShader9 *pShader) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    auto *s = static_cast<MtlPixelShader9 *>(pShader);
    if (recordingBlock_) { recordingBlock_->recordPs(s); return D3D_S_OK; }
    if (state_.ps != s) {
        if (s) s->AddRef();
        if (state_.ps) state_.ps->Release();
        state_.ps = s;
        psoDirty_ = true;
        psTexDirtyMask_ = 0xFFFFu;
        psSampDirtyMask_ = 0xFFFFu;
        boundPsSampCbvMask_ = 0;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetPixelShader(IMtlDirect3DPixelShader9 **ppShader) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!ppShader) return D3D_ERR_INVALIDCALL;
    if (state_.ps) state_.ps->AddRef();
    *ppShader = state_.ps;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetPixelShaderConstantF(uint32_t StartRegister, const float *pConstantData, uint32_t Vector4fCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordPsConstF(StartRegister, pConstantData, Vector4fCount); return D3D_S_OK; }
    if (StartRegister < 224) {
        uint32_t n = std::min(Vector4fCount, 224u - StartRegister);
        size_t bytes = size_t(n) * 16u;
        if (bytes > 0 && std::memcmp(state_.psConstF[StartRegister], pConstantData, bytes) != 0) {
            std::memcpy(state_.psConstF[StartRegister], pConstantData, bytes);
            psConstFVersion_++;
        }
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetPixelShaderConstantF(uint32_t StartRegister, float *pConstantData, uint32_t Vector4fCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (StartRegister < 224) {
        uint32_t n = std::min(Vector4fCount, 224u - StartRegister);
        std::memcpy(pConstantData, state_.psConstF[StartRegister], size_t(n) * 16u);
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetPixelShaderConstantI(uint32_t StartRegister, const int32_t *pConstantData, uint32_t Vector4iCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordPsConstI(StartRegister, pConstantData, Vector4iCount); return D3D_S_OK; }
    if (StartRegister < 16) {
        uint32_t n = std::min(Vector4iCount, 16u - StartRegister);
        std::memcpy(state_.psConstI[StartRegister], pConstantData, size_t(n) * 16u);
        psConstIDirty_ = true;
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetPixelShaderConstantI(uint32_t StartRegister, int32_t *pConstantData, uint32_t Vector4iCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (StartRegister < 16) {
        uint32_t n = std::min(Vector4iCount, 16u - StartRegister);
        std::memcpy(pConstantData, state_.psConstI[StartRegister], size_t(n) * 16u);
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::SetPixelShaderConstantB(uint32_t StartRegister, const uint32_t *pConstantData, uint32_t BoolCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    if (recordingBlock_) { recordingBlock_->recordPsConstB(StartRegister, pConstantData, BoolCount); return D3D_S_OK; }
    for (uint32_t i = 0; i < BoolCount && StartRegister + i < 16; i++) {
        state_.psConstB[StartRegister + i] = pConstantData[i] ? 1u : 0u;
    }
    psConstBDirty_ = true;
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::GetPixelShaderConstantB(uint32_t StartRegister, uint32_t *pConstantData, uint32_t BoolCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (!pConstantData) return D3D_ERR_INVALIDCALL;
    for (uint32_t i = 0; i < BoolCount && StartRegister + i < 16; i++) {
        pConstantData[i] = state_.psConstB[StartRegister + i];
    }
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::CreateQuery(uint32_t Type, IMtlDirect3DQuery9 **ppQuery) {
    if (Type != D3DM_QUERYTYPE_EVENT && Type != D3DM_QUERYTYPE_OCCLUSION) {
        return D3D_ERR_NOTAVAILABLE;
    }
    if (!ppQuery) return D3D_S_OK; /* D3D9 capability check when ppQuery == NULL */
    if (Type == D3DM_QUERYTYPE_OCCLUSION) hasAnyOcclusionQuery_ = true;
    *ppQuery = new MtlQuery9(this, Type);
    return D3D_S_OK;
}

/* ========================================================================= */
/* Queries & Command Buffer Lifecycle                                        */
/* ========================================================================= */

uint32_t MtlDirect3DDevice9::beginOcclusionQuery() {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    uint32_t slot = nextVisSlot_++ % kMaxVisibilitySlots;
    if (visPtr_) visPtr_[slot] = 0;
    activeVisSlot_ = slot;
    if (encoder_ != nil && visBuffer_ != nil) {
        if (!passHasVisBuffer_) {
            if (perfLogEnabled_) perfVisSplitsAcc_++;
            endRenderPass();
        } else {
            [encoder_ setVisibilityResultMode:MTLVisibilityResultModeCounting offset:slot * sizeof(uint64_t)];
        }
    }
    return slot;
}

uint64_t MtlDirect3DDevice9::endOcclusionQuery(uint32_t slot) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (activeVisSlot_ == slot) {
        activeVisSlot_ = kInvalidVisSlot;
        if (encoder_ != nil) {
            [encoder_ setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        }
    }
    occEndedInCmdBuf_++;
    return hasPendingWork_ ? currentCmdSeq_ : lastCommittedSeq_.load(std::memory_order_relaxed);
}

uint64_t MtlDirect3DDevice9::issueEventQuery() {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    return hasPendingWork_ ? currentCmdSeq_ : lastCommittedSeq_.load(std::memory_order_relaxed);
}

uint64_t MtlDirect3DDevice9::readVisibilityResult(uint32_t slot) const {
    if (!visPtr_ || slot >= kMaxVisibilitySlots) return 0;
    return visPtr_[slot];
}

bool MtlDirect3DDevice9::waitForCommandSeq(uint64_t targetSeq, uint32_t timeoutMs) {
    if (completedCmdSeq_.load(std::memory_order_acquire) >= targetSeq) return true;
    std::unique_lock<std::mutex> lk(seqWaitMu_);
    return seqWaitCv_.wait_for(lk, std::chrono::milliseconds(timeoutMs), [&] {
        return completedCmdSeq_.load(std::memory_order_acquire) >= targetSeq;
    });
}

void MtlDirect3DDevice9::maybeSplitAfterOcclusionBatch(uint32_t minEnded) {
    if (occSplitsThisFrame_ < 2u && occEndedInCmdBuf_ >= minEnded &&
        activeVisSlot_ == kInvalidVisSlot && cmdBuf_ != nil) {
        endRenderPass();
        flushPendingBlits();
        if (cmdBuf_ != nil) {
            commitCommandBuffer(false, false, nil, false);
            occSplitsThisFrame_++;
        }
    }
}

void MtlDirect3DDevice9::ensureCommandBuffer() {
    if (cmdBuf_ == nil) {
        cmdBuf_ = [cmdQueue_ commandBuffer];
        hasPendingWork_ = false;
    }
}

void MtlDirect3DDevice9::flushPendingBlits() {
    if (pendingUploads_.empty() && pendingMipGens_.empty()) return;
    endRenderPass();
    ensureCommandBuffer();

    id<MTLBlitCommandEncoder> blit = [cmdBuf_ blitCommandEncoder];
    if (blit) {
        for (const auto &u : pendingUploads_) {
            if (!u.srcBuffer || !u.dstTexture) continue;
            [blit copyFromBuffer:u.srcBuffer
                    sourceOffset:u.srcOffset
               sourceBytesPerRow:u.srcBytesPerRow
             sourceBytesPerImage:u.srcBytesPerImage
                      sourceSize:u.dstSize
                       toTexture:u.dstTexture
                destinationSlice:u.dstSlice
                destinationLevel:u.dstLevel
               destinationOrigin:u.dstOrigin];
        }
        for (id<MTLTexture> tex : pendingMipGens_) {
            if (tex && [tex mipmapLevelCount] > 1) {
                [blit generateMipmapsForTexture:tex];
            }
        }
        [blit endEncoding];
        hasPendingWork_ = true;
    }
    pendingUploads_.clear();
    pendingMipGens_.clear();
}

void MtlDirect3DDevice9::invalidatePassBindCache() {
    boundPso_ = nil;
    boundDss_ = nil;
    boundPsCompiled_ = nullptr;
    boundActiveRtMask_ = 0;
    boundAttachDs_ = false;
    boundPsPointSpriteEnable_ = false;
    boundPsDepthCmpMask_ = 0;
    boundPsNullMask_ = 0;
    boundPsProjMask_ = 0;
    boundPsGatherMask_ = 0;
    psoDirty_ = true;
    dssDirty_ = true;
    vpDirty_ = true;
    scissorDirty_ = true;
    cullDirty_ = true;
    fillDirty_ = true;
    slopeDirty_ = true;
    blendColorDirty_ = true;
    stencilRefDirty_ = true;
    frontFaceBound_ = false;
    zeroAttrBound_ = false;
    streamDirtyMask_ = 0xFFFFu;
    psTexDirtyMask_ = 0xFFFFu;
    psSampDirtyMask_ = 0xFFFFu;
    for (uint32_t i = 0; i < 16; i++) {
        boundVb_[i] = nil;
        boundVbOff_[i] = 0;
        boundFragTex_[i] = nil;
        boundFragSmp_[i] = nil;
    }
    boundVsConstFVecs_ = 0;
    boundPsConstFVecs_ = 0;
    vsConstIDirty_ = vsConstBDirty_ = true;
    psConstIDirty_ = psConstBDirty_ = true;
    vsFogDirty_ = psFogDirty_ = psTexStageDirty_ = true;
    boundPsSampCbvMask_ = 0;
}

static const char *shortMtlFmtName(uint32_t fmt) {
    switch (static_cast<MTLPixelFormat>(fmt)) {
        case MTLPixelFormatInvalid:          return "-";
        case MTLPixelFormatBGRA8Unorm:       return "BGRA8";
        case MTLPixelFormatBGRA8Unorm_sRGB:  return "BGRA8s";
        case MTLPixelFormatRGBA8Unorm:       return "RGBA8";
        case MTLPixelFormatRGBA8Unorm_sRGB:  return "RGBA8s";
        case MTLPixelFormatRGB10A2Unorm:     return "RGB10A2";
        case MTLPixelFormatRG11B10Float:     return "R11G11B10F";
        case MTLPixelFormatRGBA16Float:      return "RGBA16F";
        case MTLPixelFormatRGBA32Float:      return "RGBA32F";
        case MTLPixelFormatR16Float:         return "R16F";
        case MTLPixelFormatR32Float:         return "R32F";
        case MTLPixelFormatRG16Float:        return "RG16F";
        case MTLPixelFormatRG32Float:        return "RG32F";
        case MTLPixelFormatDepth32Float:     return "D32F";
        case MTLPixelFormatDepth24Unorm_Stencil8: return "D24S8";
        case MTLPixelFormatDepth32Float_Stencil8: return "D32S8";
        default:                             return "OTHER";
    }
}

void MtlDirect3DDevice9::endRenderPass(bool discardDepth, bool discardStencil, uint8_t discardColorMask) {
    bool profileThisPass = gpuPassProfileActive_ && (encoder_ != nil) && (cmdBuf_ != nil);
    if (encoder_ != nil) {
        if (passHasColorAttach_) {
            for (int i = 0; i < 4; i++) {
                if (passColorMask_ & uint8_t(1u << i)) {
                    [encoder_ setColorStoreAction:((discardColorMask & uint8_t(1u << i)) ? MTLStoreActionDontCare : MTLStoreActionStore)
                                          atIndex:NSUInteger(i)];
                }
            }
        }
        if (passHasDepthAttach_) {
            [encoder_ setDepthStoreAction:(discardDepth ? MTLStoreActionDontCare : MTLStoreActionStore)];
        }
        if (passHasStencilAttach_) {
            [encoder_ setStencilStoreAction:(discardStencil ? MTLStoreActionDontCare : MTLStoreActionStore)];
        }
        [encoder_ endEncoding];
        encoder_ = nil;
    }
    if (profileThisPass) {
        curGpuPass_.discardD = discardDepth ? 1 : 0;
        curGpuPass_.discardS = discardStencil ? 1 : 0;
        uint64_t seq = currentCmdSeq_++;
        lastCommittedSeq_.store(seq, std::memory_order_release);
        id<MTLCommandBuffer> cb = cmdBuf_;
        cmdBuf_ = nil;
        hasPendingWork_ = false;
        occEndedInCmdBuf_ = 0;
        std::atomic<uint64_t> *completedPtr = &completedCmdSeq_;
        std::mutex *waitMuPtr = &seqWaitMu_;
        std::condition_variable *waitCvPtr = &seqWaitCv_;
        [cb addCompletedHandler:^(id<MTLCommandBuffer>) {
            {
                std::lock_guard<std::mutex> lk(*waitMuPtr);
                uint64_t prev = completedPtr->load(std::memory_order_relaxed);
                while (prev < seq && !completedPtr->compare_exchange_weak(prev, seq, std::memory_order_release)) {}
            }
            waitCvPtr->notify_all();
        }];
        [cb commit];
        [cb waitUntilCompleted];
        double passMs = std::max(0.0, (cb.GPUEndTime - cb.GPUStartTime) * 1000.0);
        gpuPassTotalMs_ += passMs;
        std::vector<std::pair<uint32_t, uint64_t>> topPs;
        topPs.reserve(curGpuPass_.psCounts.size());
        for (const auto &kv : curGpuPass_.psCounts) topPs.push_back({kv.second, kv.first});
        std::sort(topPs.begin(), topPs.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        fprintf(stderr, "GPU_PASS #%02u: %ux%u rt0=%s(%s) rt1=%s ds=%s(%s/%s->%s/%s) draws=%u gpu=%.2f ms",
                gpuPassIndex_++, curGpuPass_.w, curGpuPass_.h,
                shortMtlFmtName(curGpuPass_.rtFmt[0]),
                curGpuPass_.rtClear[0] == 1 ? "CLR" : (curGpuPass_.rtClear[0] == 2 ? "DONT" : "LOAD"),
                shortMtlFmtName(curGpuPass_.rtFmt[1]),
                shortMtlFmtName(curGpuPass_.dsFmt),
                curGpuPass_.zClear ? "CLR" : "LOAD",
                curGpuPass_.sClear ? "CLR" : "LOAD",
                curGpuPass_.discardD ? "DISC" : "STOR",
                curGpuPass_.discardS ? "DISC" : "STOR",
                curGpuPass_.draws, passMs);
        for (size_t k = 0; k < std::min<size_t>(topPs.size(), 3); k++) {
            fprintf(stderr, " %sps=%016llx(%u)", k == 0 ? "top:[" : ",",
                    (unsigned long long)topPs[k].second, topPs[k].first);
        }
        if (!topPs.empty()) fprintf(stderr, "]");
        fprintf(stderr, "\n");
    }
    passHasColorAttach_ = false;
    passColorMask_ = 0;
    pendingColorDontCare_ = false;
    passHasDepthAttach_ = false;
    passHasStencilAttach_ = false;
    passHasVisBuffer_ = false;
    for (int i = 0; i < 4; i++) passColorSurfaces_[i] = nullptr;
    passDsSurface_ = nullptr;
    invalidatePassBindCache();
}

uint8_t MtlDirect3DDevice9::activeColorWriteRtMask() const {
    static const uint32_t kWriteMaskRs[4] = {
        D3DM_RS_COLORWRITEENABLE,
        D3DM_RS_COLORWRITEENABLE1,
        D3DM_RS_COLORWRITEENABLE2,
        D3DM_RS_COLORWRITEENABLE3,
    };
    uint8_t mask = 0;
    for (int i = 0; i < 4; i++) {
        MtlSurface9 *rt = renderTargets_[i];
        if (!rt || rt->isNull() || !rt->texture()) continue;
        if ((state_.renderStates[kWriteMaskRs[i]] & 0xFu) != 0) {
            mask |= uint8_t(1u << i);
        }
    }
    return mask;
}

bool MtlDirect3DDevice9::beginRenderPassIfNeeded(bool allowColorForDraw) {
    bool wantSrgb = (state_.renderStates[D3DM_RS_SRGBWRITEENABLE] != 0);

    /* If uploads arrived while a pass is open, flush them before drawing */
    if (!pendingUploads_.empty() || !pendingMipGens_.empty()) {
        flushPendingBlits();
    }

    /* If SRGBWRITEENABLE toggled while a pass is open on an sRGB-capable RT, reopen pass */
    if (encoder_ != nil && wantSrgb != passSrgbWrite_) {
        bool anySrgbCapable = false;
        for (int i = 0; i < 4; i++) {
            if (renderTargets_[i] && !renderTargets_[i]->isNull() &&
                renderTargets_[i]->formatInfo().srgbPixelFormat != MTLPixelFormatInvalid) {
                anySrgbCapable = true;
                break;
            }
        }
        if (anySrgbCapable && encoder_ != nil) endRenderPass();
    }

    if (encoder_ != nil) return true;

    /* Compute active pass dimensions */
    uint32_t rtW = 0, rtH = 0;
    for (int i = 0; i < 4; i++) {
        if (renderTargets_[i]) {
            rtW = renderTargets_[i]->desc().Width;
            rtH = renderTargets_[i]->desc().Height;
            break;
        }
    }
    if (rtW == 0 && depthStencil_) {
        rtW = depthStencil_->desc().Width;
        rtH = depthStencil_->desc().Height;
    }
    if (rtW == 0 || rtH == 0) return false;

    bool attachDs = (depthStencil_ != nullptr && depthStencil_->texture() != nil &&
                     depthStencil_->desc().Width >= rtW && depthStencil_->desc().Height >= rtH);

    ensureCommandBuffer();
    MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.renderTargetWidth  = rtW;
    rpd.renderTargetHeight = rtH;
    bool needVisBuf = (visBuffer_ != nil) &&
                      (activeVisSlot_ != kInvalidVisSlot || (hasAnyOcclusionQuery_ && attachDs));
    rpd.visibilityResultBuffer = needVisBuf ? visBuffer_ : nil;

    bool clearZ = depthStencil_ ? depthStencil_->hasPendingZClear() : false;
    bool clearS = depthStencil_ ? (depthStencil_->formatInfo().hasStencil && depthStencil_->hasPendingSClear()) : false;
    float zVal = depthStencil_ ? depthStencil_->pendingClearZ() : 1.0f;
    uint32_t sVal = depthStencil_ ? depthStencil_->pendingClearStencil() : 0u;
    bool attachColor = !attachDs || (allowColorForDraw && (activeColorWriteRtMask() != 0 || (!clearZ && !clearS)));

    bool anyAttach = false;
    bool anyColorAttached = false;
    uint8_t attachedColorMask = 0;
    uint8_t consumedColorClearMask = 0;
    if (attachColor) {
        for (int i = 0; i < 4; i++) {
            MtlSurface9 *rt = renderTargets_[i];
            if (!rt || rt->isNull() || !rt->texture()) continue;
            if (rt->desc().Width < rtW || rt->desc().Height < rtH) continue;

            bool useSrgbView = wantSrgb && (rt->formatInfo().srgbPixelFormat != MTLPixelFormatInvalid);
            bool clrRt = rt->hasPendingColorClear();
            rpd.colorAttachments[i].texture     = useSrgbView ? rt->srgbRenderTargetView() : rt->texture();
            rpd.colorAttachments[i].level       = rt->mipLevel();
            rpd.colorAttachments[i].slice       = rt->slice();
            rpd.colorAttachments[i].loadAction  = clrRt
                ? MTLLoadActionClear
                : ((i == 0 && pendingColorDontCare_) ? MTLLoadActionDontCare : MTLLoadActionLoad);
            rpd.colorAttachments[i].storeAction = MTLStoreActionUnknown;
            if (clrRt) {
                D3DCOLOR_T clrVal = rt->pendingClearColor();
                uint8_t cr = uint8_t((clrVal >> 16) & 0xFFu);
                uint8_t cg = uint8_t((clrVal >>  8) & 0xFFu);
                uint8_t cb = uint8_t((clrVal >>  0) & 0xFFu);
                uint8_t ca = uint8_t((clrVal >> 24) & 0xFFu);
                if (useSrgbView) {
                    /* D3D9 Clear writes literal channel values without sRGB encoding */
                    rpd.colorAttachments[i].clearColor = MTLClearColorMake(
                        srgbByteToLinear(cr), srgbByteToLinear(cg), srgbByteToLinear(cb), double(ca) / 255.0);
                } else {
                    rpd.colorAttachments[i].clearColor = MTLClearColorMake(
                        double(cr) / 255.0, double(cg) / 255.0, double(cb) / 255.0, double(ca) / 255.0);
                }
                consumedColorClearMask |= uint8_t(1u << i);
            }
            passColorSurfaces_[i] = rt;
            anyAttach = true;
            anyColorAttached = true;
            attachedColorMask |= uint8_t(1u << i);
        }
    }
    pendingColorDontCare_ = false;

    if (attachDs) {
        rpd.depthAttachment.texture     = depthStencil_->texture();
        rpd.depthAttachment.level       = depthStencil_->mipLevel();
        rpd.depthAttachment.slice       = depthStencil_->slice();
        rpd.depthAttachment.loadAction  = clearZ ? MTLLoadActionClear : MTLLoadActionLoad;
        rpd.depthAttachment.storeAction = MTLStoreActionUnknown;
        rpd.depthAttachment.clearDepth  = double(zVal);

        if (depthStencil_->formatInfo().hasStencil) {
            rpd.stencilAttachment.texture      = depthStencil_->texture();
            rpd.stencilAttachment.level        = depthStencil_->mipLevel();
            rpd.stencilAttachment.slice        = depthStencil_->slice();
            rpd.stencilAttachment.loadAction   = clearS ? MTLLoadActionClear : MTLLoadActionLoad;
            rpd.stencilAttachment.storeAction  = MTLStoreActionUnknown;
            rpd.stencilAttachment.clearStencil = sVal & 0xFFu;
        }
        anyAttach = true;
    }

    if (!anyAttach) return false;

    encoder_ = [cmdBuf_ renderCommandEncoderWithDescriptor:rpd];
    if (!encoder_) {
        for (int i = 0; i < 4; i++) passColorSurfaces_[i] = nullptr;
        return false;
    }

    for (int i = 0; i < 4; i++) {
        if (consumedColorClearMask & uint8_t(1u << i)) {
            MtlSurface9 *rt = passColorSurfaces_[i];
            if (rt) {
                rt->clearPendingColorClear();
                if (!rt->hasAnyPendingClear()) unregisterPendingClearSurface(rt);
            }
        }
    }
    if (attachDs && depthStencil_) {
        if (clearZ) depthStencil_->clearPendingZClear();
        if (clearS) depthStencil_->clearPendingSClear();
        if (!depthStencil_->hasAnyPendingClear()) unregisterPendingClearSurface(depthStencil_);
    }

    passSrgbWrite_        = wantSrgb;
    passWidth_            = rtW;
    passHeight_           = rtH;
    passHasColorAttach_   = anyColorAttached;
    passColorMask_        = attachedColorMask;
    passHasDepthAttach_   = attachDs;
    passHasStencilAttach_ = attachDs && depthStencil_->formatInfo().hasStencil;
    passHasVisBuffer_     = needVisBuf;
    passDsSurface_        = attachDs ? depthStencil_ : nullptr;
    hasPendingWork_       = true;

    if (activeVisSlot_ != kInvalidVisSlot && visBuffer_ != nil) {
        [encoder_ setVisibilityResultMode:MTLVisibilityResultModeCounting offset:activeVisSlot_ * sizeof(uint64_t)];
    }
    if (gpuPassProfileActive_) {
        curGpuPass_.w = rtW;
        curGpuPass_.h = rtH;
        for (int i = 0; i < 4; i++) {
            curGpuPass_.rtFmt[i] = uint32_t(rpd.colorAttachments[i].texture ? [rpd.colorAttachments[i].texture pixelFormat] : MTLPixelFormatInvalid);
            MTLLoadAction la = rpd.colorAttachments[i].loadAction;
            curGpuPass_.rtClear[i] = (la == MTLLoadActionClear) ? 1 : (la == MTLLoadActionDontCare ? 2 : 0);
        }
        curGpuPass_.dsFmt = uint32_t(attachDs && rpd.depthAttachment.texture ? [rpd.depthAttachment.texture pixelFormat] : MTLPixelFormatInvalid);
        curGpuPass_.zClear = clearZ ? 1 : 0;
        curGpuPass_.sClear = clearS ? 1 : 0;
        curGpuPass_.draws = 0;
        curGpuPass_.psCounts.clear();
    }
    return true;
}

void MtlDirect3DDevice9::commitCommandBuffer(bool waitUntilCompleted, bool isPresentFrame,
                                             id<CAMetalDrawable> drawableToPresent, bool useVsync) {
    (void)useVsync;
    endRenderPass();

    if (cmdBuf_ == nil) {
        if (isPresentFrame) {
            /* No GPU work this frame; release frame slot semaphore immediately */
            dispatch_semaphore_signal(frameSem_);
        }
        return;
    }

    uint64_t seq = currentCmdSeq_++;
    lastCommittedSeq_.store(seq, std::memory_order_release);
    if (perfLogEnabled_) perfCommitsAcc_++;
    id<MTLCommandBuffer> cb = cmdBuf_;
    cmdBuf_ = nil;
    hasPendingWork_ = false;
    occEndedInCmdBuf_ = 0;

    dispatch_semaphore_t sem = isPresentFrame ? frameSem_ : nullptr;
    std::atomic<uint64_t> *completedPtr = &completedCmdSeq_;
    std::mutex *waitMuPtr = &seqWaitMu_;
    std::condition_variable *waitCvPtr = &seqWaitCv_;
    std::atomic<uint64_t> *gpuUsPtr = &perfGpuUsTotal_;
    std::atomic<uint32_t> *gpuSamplesPtr = &perfGpuSamples_;
    const bool collectPerf = perfLogEnabled_;

    [cb addCompletedHandler:^(id<MTLCommandBuffer> doneCb) {
        {
            std::lock_guard<std::mutex> lk(*waitMuPtr);
            uint64_t prev = completedPtr->load(std::memory_order_relaxed);
            while (prev < seq && !completedPtr->compare_exchange_weak(prev, seq, std::memory_order_release)) {}
        }
        waitCvPtr->notify_all();
        if (collectPerf) {
            double dt = doneCb.GPUEndTime - doneCb.GPUStartTime;
            if (dt > 0.0 && dt < 1.0) {
                gpuUsPtr->fetch_add(uint64_t(dt * 1e6), std::memory_order_relaxed);
            }
            if (isPresentFrame) {
                gpuSamplesPtr->fetch_add(1u, std::memory_order_relaxed);
            }
        }
        if (sem) dispatch_semaphore_signal(sem);
    }];

    if (drawableToPresent != nil) {
        [cb presentDrawable:drawableToPresent];
    }

    [cb commit];
    if (waitUntilCompleted) {
        [cb waitUntilCompleted];
    }
}

void MtlDirect3DDevice9::flushCommands(bool waitUntilCompleted) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    endRenderPass();
    flushPendingBlits();
    if (cmdBuf_ != nil) {
        commitCommandBuffer(waitUntilCompleted, false, nil, false);
    }
}

MTL_HRESULT MtlDirect3DDevice9::presentSwapChain(MtlSwapChain9 *sc, const D3DMtlRect *srcRect,
                                                 const D3DMtlRect *dstRect, uint32_t flags) {
    (void)flags;
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);

    bool discAutoDs = (passDsSurface_ != nullptr && passDsSurface_ == autoDepthStencil_);
    endRenderPass(discAutoDs, discAutoDs);
    MtlSurface9 *bb = sc->backBuffer();
    if (bb && bb->hasAnyPendingClear()) {
        flushSurfacePendingClear(bb);
    }
    flushPendingBlits();

    @autoreleasepool {
        auto t0 = perfLogEnabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        CAMetalLayer *layer = sc->metalLayer();
        id<CAMetalDrawable> drawable = (layer != nil) ? [layer nextDrawable] : nil;
        auto tDrawable = perfLogEnabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        uint32_t outW = 0, outH = 0;

        if (drawable != nil && bb != nil && bb->texture() != nil) {
            ensureCommandBuffer();
            id<MTLTexture> dstTex = drawable.texture;
            uint32_t srcW = bb->desc().Width;
            uint32_t srcH = bb->desc().Height;
            uint32_t dstW = uint32_t([dstTex width]);
            uint32_t dstH = uint32_t([dstTex height]);
            outW = dstW; outH = dstH;

            if (!srcRect && !dstRect && srcW == dstW && srcH == dstH &&
                bb->formatInfo().pixelFormat == [dstTex pixelFormat]) {
                id<MTLBlitCommandEncoder> blit = [cmdBuf_ blitCommandEncoder];
                [blit copyFromTexture:bb->texture() sourceSlice:0 sourceLevel:0
                         sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(srcW, srcH, 1)
                            toTexture:dstTex destinationSlice:0 destinationLevel:0
                    destinationOrigin:MTLOriginMake(0, 0, 0)];
                [blit endEncoding];
                hasPendingWork_ = true;
            } else {
                blitQuad(bb->texture(), srcRect, srcW, srcH,
                         dstTex, 0, 0, [dstTex pixelFormat], dstRect, dstW, dstH, true);
            }
        }

        bool useVsync = (sc->presentParams().PresentationInterval != D3DM_PRESENT_INTERVAL_IMMEDIATE);
        commitCommandBuffer(false, true, drawable, useVsync);
        occSplitsThisFrame_ = 0;
        if (pipelineCache_) pipelineCache_->maybeFlushArchive();
        if (gpuPassLogEnabled_) {
            gpuPassFrameCounter_++;
            if (gpuPassProfileActive_) {
                fprintf(stderr, "GPU_PASS SUMMARY frame=%llu passes=%u serializedTotalGpu=%.2f ms\n",
                        (unsigned long long)gpuPassFrameCounter_, gpuPassIndex_, gpuPassTotalMs_);
                gpuPassProfileActive_ = false;
            } else if (gpuPassFrameCounter_ == 2800u) {
                flushCommands(true);
                gpuPassProfileActive_ = true;
                gpuPassIndex_ = 0;
                gpuPassTotalMs_ = 0.0;
                fprintf(stderr, "=== GPU_PASS SNAPSHOT BEGIN (frame %llu, %ux%u) ===\n",
                        (unsigned long long)(gpuPassFrameCounter_ + 1u),
                        bb ? bb->desc().Width : 0u, bb ? bb->desc().Height : 0u);
            }
        }
        auto tDone = perfLogEnabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        /* Advance triple-buffered ring buffer to next in-flight frame slot */
        frameIndex_++;
        dispatch_semaphore_wait(frameSem_, DISPATCH_TIME_FOREVER);
        auto tWaitDone = perfLogEnabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        ringBuffer_->beginFrame(frameIndex_);

        if (perfLogEnabled_) {
            double waitMs = std::chrono::duration<double, std::milli>((tDrawable - t0) + (tWaitDone - tDone)).count();
            double presentMs = std::chrono::duration<double, std::milli>(tDone - tDrawable).count();
            perfWaitMsAcc_ += waitMs;
            perfCpuMsAcc_ += presentMs;
            perfFrames_++;
            if ((perfFrames_ % 60) == 0) {
                uint32_t gCnt   = perfGpuSamples_.exchange(0, std::memory_order_relaxed);
                uint64_t gUs    = perfGpuUsTotal_.exchange(0, std::memory_order_relaxed);
                uint64_t qNs    = perfQueryWaitNsAcc_.exchange(0, std::memory_order_relaxed);
                uint64_t qOccNs = perfOccQueryWaitNsAcc_.exchange(0, std::memory_order_relaxed);
                uint64_t qEvtNs = perfEvtQueryWaitNsAcc_.exchange(0, std::memory_order_relaxed);
                uint32_t qFl    = perfQueryFlushFlagAcc_.exchange(0, std::memory_order_relaxed);
                uint32_t qTr    = perfQueryTrigFlushAcc_.exchange(0, std::memory_order_relaxed);
                uint32_t qMs    = perfQueryMissAcc_.exchange(0, std::memory_order_relaxed);
                double avgGpuMs = gCnt ? (double(gUs) / (1000.0 * gCnt)) : 0.0;
                double avgDraws = double(perfDrawsAcc_) / 60.0;
                fprintf(stderr,
                        "RunnerMetal PERF: bb=%ux%u drawable=%ux%u draws=%.1f commits=%.1f visSplits=%.1f "
                        "gpu=%.2f ms/frame presentCpu=%.2f waitCpu=%.2f qCpu=%.2f (occ=%.2f evt=%.2f) ms/frame "
                        "(qMiss=%.0f qFlushFlag=%.0f qTrigFlush=%.0f)\n",
                        bb ? bb->desc().Width : 0u, bb ? bb->desc().Height : 0u,
                        outW, outH, avgDraws, double(perfCommitsAcc_) / 60.0, double(perfVisSplitsAcc_) / 60.0,
                        avgGpuMs, perfCpuMsAcc_ / 60.0, perfWaitMsAcc_ / 60.0,
                        double(qNs) / 60000000.0, double(qOccNs) / 60000000.0, double(qEvtNs) / 60000000.0,
                        double(qMs) / 60.0, double(qFl) / 60.0, double(qTr) / 60.0);
                perfCpuMsAcc_ = 0.0;
                perfWaitMsAcc_ = 0.0;
                perfDrawsAcc_ = 0;
                perfCommitsAcc_ = 0;
                perfVisSplitsAcc_ = 0;
            }
        }
        if (stallLogEnabled_) {
            auto now = std::chrono::steady_clock::now();
            double dtMs = hasLastPresentTime_
                ? std::chrono::duration<double, std::milli>(now - lastPresentTime_).count()
                : 0.0;
            lastPresentTime_ = now;
            hasLastPresentTime_ = true;

            uint32_t vsCnt = stallVsCreateCnt_.exchange(0u, std::memory_order_relaxed);
            double   vsMs  = double(stallVsCreateNs_.exchange(0u, std::memory_order_relaxed)) / 1e6;
            uint32_t psCnt = stallPsCreateCnt_.exchange(0u, std::memory_order_relaxed);
            double   psMs  = double(stallPsCreateNs_.exchange(0u, std::memory_order_relaxed)) / 1e6;
            double   compilePsoMs = vsMs + psMs + stallPsVarMissMs_ + stallPsoMissMs_;
            if (dtMs > 25.0 || compilePsoMs > 5.0 || stallBgPsHitCnt_ > 0 || stallWarmPsoHitCnt_ > 0) {
                fprintf(stderr,
                        "STALL_LOG: frame %u dt=%.1f ms | vs=%u(%.1f) psParse=%u(%.1f) "
                        "psVar=%u(%.1f) psoHit=%u(%.1f) psoMiss=%u(%.1f) | bgPsReady=%u warmPsoReady=%u\n",
                        frameIndex_, dtMs, vsCnt, vsMs, psCnt, psMs,
                        stallPsVarMissCnt_, stallPsVarMissMs_,
                        stallPsoHitCnt_, stallPsoHitMs_,
                        stallPsoMissCnt_, stallPsoMissMs_,
                        stallBgPsHitCnt_, stallWarmPsoHitCnt_);
            }
            stallPsVarMissCnt_ = 0; stallPsVarMissMs_ = 0.0;
            stallPsoHitCnt_ = 0;    stallPsoHitMs_ = 0.0;
            stallPsoMissCnt_ = 0;   stallPsoMissMs_ = 0.0;
            stallBgPsHitCnt_ = 0;   stallWarmPsoHitCnt_ = 0;
        }
        return D3D_S_OK;
    }
}

/* ========================================================================= */
/* Draw Preparation & Draw Calls                                             */
/* ========================================================================= */

bool MtlDirect3DDevice9::prepareDraw(uint32_t primitiveType, bool useUpStream0, id<MTLBuffer> upVb, size_t upVbOffset,
                                     uint32_t upStride, const MtlCompiledShader **outPs) {
    if (!state_.vs || !state_.vs->parsed()) return false;
    MtlParsedShader *vsParsed = state_.vs->parsed();

    MtlParsedShader *psParsed = nullptr;
    if (state_.ps) {
        if (stallLogEnabled_ && !state_.ps->hasReadyParsed()) {
            auto tPsResolve = std::chrono::steady_clock::now();
            psParsed = state_.ps->parsed();
            uint64_t ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tPsResolve).count());
            stallPsCreateNs_.fetch_add(ns, std::memory_order_relaxed);
        } else {
            psParsed = state_.ps->parsed();
        }
    }

    if (!pendingClearSurfaces_.empty()) {
        uint32_t vsMask = vsParsed->defaultCompiled.usedSamplerMask;
        uint32_t psMask = psParsed ? psParsed->psDeclaredSamplerMask : 0u;
        flushPendingClearsForSampledTextures(vsMask, psMask);
    }

    if (encoder_ != nil && passHasDepthAttach_ && !passHasColorAttach_ && activeColorWriteRtMask() != 0) {
        endRenderPass();
    }
    if (!beginRenderPassIfNeeded(true)) return false;
    if (perfLogEnabled_) perfDrawsAcc_++;

    /* Clamp viewport */
    uint32_t vpX = std::min(state_.viewport.X, passWidth_);
    uint32_t vpY = std::min(state_.viewport.Y, passHeight_);
    uint32_t vpW = std::min(state_.viewport.Width,  passWidth_ - vpX);
    uint32_t vpH = std::min(state_.viewport.Height, passHeight_ - vpY);
    if (vpW == 0 || vpH == 0) return false;

    /* Clamp scissor */
    uint32_t scX = 0, scY = 0, scW = passWidth_, scH = passHeight_;
    if (state_.renderStates[D3DM_RS_SCISSORTESTENABLE]) {
        int32_t l = std::clamp(state_.scissorRect.left,   0, int32_t(passWidth_));
        int32_t t = std::clamp(state_.scissorRect.top,    0, int32_t(passHeight_));
        int32_t r = std::clamp(state_.scissorRect.right,  l, int32_t(passWidth_));
        int32_t b = std::clamp(state_.scissorRect.bottom, t, int32_t(passHeight_));
        if (r <= l || b <= t) return false;
        scX = uint32_t(l); scY = uint32_t(t); scW = uint32_t(r - l); scH = uint32_t(b - t);
    }

    const MtlCompiledShader *psCompiled = boundPsCompiled_;

    if (useUpStream0) {
        psoDirty_ = true;
        streamDirtyMask_ |= 1u;
    }

    /* Compute active PS sampler specialization masks (~1-4 loop iterations) */
    uint16_t psDepthCmpMask = 0, psNullMask = 0, psProjMask = 0, psGatherMask = 0;
    if (psParsed) {
        uint16_t mask = psParsed->psDeclaredSamplerMask;
        while (mask) {
            uint32_t s = uint32_t(__builtin_ctz(mask));
            mask &= uint16_t(mask - 1u);
            MtlSamplerTexKind expectedKind = psParsed->psDeclaredSamplerKind[s];
            MtlTextureCommon *tc = getTextureCommon(state_.textures[s]);
            bool valid = tc && tc->baseTexture != nil &&
                ((expectedKind == MtlSamplerTexKind::Tex2D   && tc->mtlType == MTLTextureType2D) ||
                 (expectedKind == MtlSamplerTexKind::TexCube && tc->mtlType == MTLTextureTypeCube) ||
                 (expectedKind == MtlSamplerTexKind::Tex3D   && tc->mtlType == MTLTextureType3D));
            uint16_t bit = uint16_t(1u << s);
            if (!valid) {
                psNullMask |= bit;
            } else {
                if (tc->fmtInfo.isDepth && !tc->fmtInfo.isRawDepthFetch) psDepthCmpMask |= bit;
                if (s < 8u && (state_.texStageStates[s][24] & 256u) != 0) psProjMask |= bit;
                if ((state_.fetch4Mask & bit) != 0) psGatherMask |= bit;
            }
        }
    }

    bool pointSpriteEnable = (primitiveType == D3DM_PT_POINTLIST) && (state_.renderStates[D3DM_RS_POINTSPRITEENABLE] != 0);

    if (psoDirty_ || boundPso_ == nil ||
        pointSpriteEnable != boundPsPointSpriteEnable_ ||
        psDepthCmpMask != boundPsDepthCmpMask_ || psNullMask != boundPsNullMask_ ||
        psProjMask != boundPsProjMask_ || psGatherMask != boundPsGatherMask_) {
        bool attachDs = (depthStencil_ != nullptr && depthStencil_->texture() != nil &&
                         depthStencil_->desc().Width >= passWidth_ && depthStencil_->desc().Height >= passHeight_);

        uint8_t activeRtMask = 0;
        MtlRenderPipelineKey psoKey = {};
        if (passHasColorAttach_) {
            for (int i = 0; i < 4; i++) {
                if (!(passColorMask_ & uint8_t(1u << i))) continue;
                MtlSurface9 *rt = renderTargets_[i];
                if (!rt || rt->isNull() || !rt->texture()) continue;
                if (rt->desc().Width < passWidth_ || rt->desc().Height < passHeight_) continue;

                activeRtMask |= uint8_t(1u << i);
                bool useSrgb = passSrgbWrite_ && (rt->formatInfo().srgbPixelFormat != MTLPixelFormatInvalid);
                psoKey.colorFormats[i] = uint16_t(useSrgb ? rt->formatInfo().srgbPixelFormat : rt->formatInfo().pixelFormat);
                if (rt->formatInfo().hasAlpha) psoKey.colorHasAlphaMask |= uint8_t(1u << i);
            }
        }

        psoKey.colorWriteMasks[0] = uint8_t(state_.renderStates[D3DM_RS_COLORWRITEENABLE]  & 0xFu);
        psoKey.colorWriteMasks[1] = uint8_t(state_.renderStates[D3DM_RS_COLORWRITEENABLE1] & 0xFu);
        psoKey.colorWriteMasks[2] = uint8_t(state_.renderStates[D3DM_RS_COLORWRITEENABLE2] & 0xFu);
        psoKey.colorWriteMasks[3] = uint8_t(state_.renderStates[D3DM_RS_COLORWRITEENABLE3] & 0xFu);

        if (attachDs) {
            psoKey.depthFormat = uint16_t(depthStencil_->formatInfo().pixelFormat);
            if (depthStencil_->formatInfo().hasStencil) {
                psoKey.stencilFormat = uint16_t(depthStencil_->formatInfo().pixelFormat);
            }
        }

        psCompiled = nullptr;
        MtlShaderVariantKey psVarKey = {};
        if (psParsed) {
            bool alphaTestEnable = (state_.renderStates[D3DM_RS_ALPHATESTENABLE] != 0);
            bool needFrag = (activeRtMask != 0) || alphaTestEnable || psParsed->psUsesDiscard || psParsed->psWritesDepth;
            if (needFrag) {
                bool flatShade = (state_.renderStates[D3DM_RS_SHADEMODE] == 1);
                uint16_t usedMask = psParsed->psDeclaredSamplerMask;
                psVarKey.vsIoHash            = vsParsed->vsOutputIoHash;
                psVarKey.colorWriteMask      = activeRtMask;
                psVarKey.hasDepthAttach      = attachDs;
                psVarKey.flatShade           = flatShade;
                psVarKey.alphaTestEnable     = alphaTestEnable;
                psVarKey.pointSpriteEnable   = pointSpriteEnable;
                psVarKey.samplerDepthCmpMask = psDepthCmpMask & usedMask;
                psVarKey.samplerNullMask     = psNullMask & usedMask;
                psVarKey.samplerProjMask     = psProjMask & usedMask;
                psVarKey.samplerGatherMask   = psGatherMask & usedMask;

                bool psVarMiss = false, bgPsHit = false;
                auto tPs0 = stallLogEnabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                psCompiled = shaderCompiler_->getOrCompilePsVariant(*psParsed, vsParsed,
                                                                    activeRtMask, attachDs, flatShade,
                                                                    alphaTestEnable, pointSpriteEnable,
                                                                    psDepthCmpMask, psNullMask,
                                                                    psProjMask, psGatherMask,
                                                                    false, &psVarMiss, &bgPsHit);
                if (stallLogEnabled_) {
                    double psMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tPs0).count();
                    if (psVarMiss) {
                        stallPsVarMissCnt_++;
                        stallPsVarMissMs_ += psMs;
                    }
                    if (bgPsHit) {
                        stallBgPsHitCnt_++;
                    }
                }
                if (!psCompiled) return false;
            }
        }

        MtlStreamBindingInfo streamInfo[16] = {};
        for (int i = 0; i < 16; i++) {
            if (i == 0 && useUpStream0) {
                streamInfo[0].bound       = (upVb != nil);
                streamInfo[0].stride      = upStride;
                streamInfo[0].freqSetting = 1;
            } else {
                streamInfo[i].bound       = (state_.streams[i].vb != nullptr);
                streamInfo[i].stride      = state_.streams[i].stride;
                streamInfo[i].freqSetting = state_.streams[i].freq;
            }
        }

        psoKey.vsHash           = vsParsed->defaultCompiled.shaderHash;
        psoKey.psHash           = psCompiled ? psCompiled->shaderHash : 0;
        psoKey.declHash         = state_.vdecl ? state_.vdecl->hash() : 0;
        psoKey.streamLayoutHash = hashStreamLayout(streamInfo);
        psoKey.sampleCount      = 1;
        psoKey.blendEnable      = state_.renderStates[D3DM_RS_ALPHABLENDENABLE] ? 1 : 0;
        psoKey.sepAlphaEnable   = state_.renderStates[D3DM_RS_SEPARATEALPHABLENDENABLE] ? 1 : 0;
        psoKey.alphaToCoverage  = (state_.renderStates[D3DM_RS_ADAPTIVETESS_Y] == D3D_MAKEFOURCC('A', 'T', 'O', 'C')) ? 1 : 0;
        psoKey.srcBlend         = uint8_t(state_.renderStates[D3DM_RS_SRCBLEND]);
        psoKey.dstBlend         = uint8_t(state_.renderStates[D3DM_RS_DESTBLEND]);
        psoKey.blendOp          = uint8_t(state_.renderStates[D3DM_RS_BLENDOP]);
        psoKey.srcBlendAlpha    = uint8_t(state_.renderStates[D3DM_RS_SRCBLENDALPHA]);
        psoKey.dstBlendAlpha    = uint8_t(state_.renderStates[D3DM_RS_DESTBLENDALPHA]);
        psoKey.blendOpAlpha     = uint8_t(state_.renderStates[D3DM_RS_BLENDOPALPHA]);

        bool psoCacheHit = false, warmPsoHit = false;
        auto tPso0 = stallLogEnabled_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        id<MTLRenderPipelineState> pso = pipelineCache_->getOrCreateRenderPipeline(
            psoKey, *vsParsed, psCompiled, state_.vdecl, streamInfo,
            stallLogEnabled_ ? &psoCacheHit : nullptr,
            stallLogEnabled_ ? &warmPsoHit : nullptr);
        if (stallLogEnabled_) {
            double psoMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tPso0).count();
            if (psoCacheHit) {
                stallPsoHitCnt_++;
                stallPsoHitMs_ += psoMs;
                if (warmPsoHit) stallWarmPsoHitCnt_++;
            } else {
                stallPsoMissCnt_++;
                stallPsoMissMs_ += psoMs;
            }
        }
        if (!pso) return false;

        uint64_t curMapHash = currentMapHash_.load(std::memory_order_relaxed);
        if (curMapHash != 0 && !useUpStream0) {
            if (recordedManifestMapHash_ != curMapHash) {
                recordedManifestMapHash_ = curMapHash;
                recordedManifestPsoKeys_.clear();
            }
            if (recordedManifestPsoKeys_.insert(psoKey).second) {
                uint64_t psBHash = (psCompiled && psParsed) ? psParsed->bytecodeHash : 0;
                pipelineCache_->recordManifestEntry(
                    curMapHash, vsParsed->bytecodeHash, psBHash,
                    psVarKey, psoKey, state_.vdecl, streamInfo);
            }
        }

        if (pso != boundPso_) {
            [encoder_ setRenderPipelineState:pso];
            boundPso_ = pso;
        }
        boundPsCompiled_          = psCompiled;
        boundAttachDs_            = attachDs;
        boundPsPointSpriteEnable_ = pointSpriteEnable;
        boundPsDepthCmpMask_      = psDepthCmpMask;
        boundPsNullMask_     = psNullMask;
        boundPsProjMask_     = psProjMask;
        boundPsGatherMask_   = psGatherMask;
        psoDirty_ = useUpStream0; /* Draw*UP uses temporary stream 0 stride; force re-check next draw */
    }
    if (outPs) *outPs = psCompiled;
    if (gpuPassProfileActive_) {
        curGpuPass_.draws++;
        curGpuPass_.psCounts[psCompiled ? psCompiled->shaderHash : 0ull]++;
    }

    if (dssDirty_ || boundDss_ == nil) {
        bool attachDs = (depthStencil_ != nullptr && depthStencil_->texture() != nil &&
                         depthStencil_->desc().Width >= passWidth_ && depthStencil_->desc().Height >= passHeight_);
        MtlDepthStencilKey dsk = {};
        dsk.hasDepth      = attachDs ? 1 : 0;
        dsk.hasStencil    = (attachDs && depthStencil_->formatInfo().hasStencil) ? 1 : 0;
        dsk.zEnable       = state_.renderStates[D3DM_RS_ZENABLE] ? 1 : 0;
        dsk.zWrite        = state_.renderStates[D3DM_RS_ZWRITEENABLE] ? 1 : 0;
        dsk.zFunc         = uint8_t(state_.renderStates[D3DM_RS_ZFUNC]);
        dsk.stencilEnable = state_.renderStates[D3DM_RS_STENCILENABLE] ? 1 : 0;
        dsk.twoSided      = state_.renderStates[D3DM_RS_TWOSIDEDSTENCILMODE] ? 1 : 0;
        dsk.frontFunc     = uint8_t(state_.renderStates[D3DM_RS_STENCILFUNC]);
        dsk.frontFail     = uint8_t(state_.renderStates[D3DM_RS_STENCILFAIL]);
        dsk.frontZFail    = uint8_t(state_.renderStates[D3DM_RS_STENCILZFAIL]);
        dsk.frontPass     = uint8_t(state_.renderStates[D3DM_RS_STENCILPASS]);
        dsk.backFunc      = uint8_t(state_.renderStates[D3DM_RS_CCW_STENCILFUNC]);
        dsk.backFail      = uint8_t(state_.renderStates[D3DM_RS_CCW_STENCILFAIL]);
        dsk.backZFail     = uint8_t(state_.renderStates[D3DM_RS_CCW_STENCILZFAIL]);
        dsk.backPass      = uint8_t(state_.renderStates[D3DM_RS_CCW_STENCILPASS]);
        dsk.readMask      = uint8_t(state_.renderStates[D3DM_RS_STENCILMASK] & 0xFFu);
        dsk.writeMask     = uint8_t(state_.renderStates[D3DM_RS_STENCILWRITEMASK] & 0xFFu);
        id<MTLDepthStencilState> dss = pipelineCache_->getOrCreateDepthStencilState(dsk);
        if (dss != boundDss_) {
            [encoder_ setDepthStencilState:dss];
            boundDss_ = dss;
        }
        dssDirty_ = false;
    }

    if (vpDirty_) {
        [encoder_ setViewport:{ double(vpX), double(vpY), double(vpW), double(vpH),
                                double(state_.viewport.MinZ), double(state_.viewport.MaxZ) }];
        vpDirty_ = false;
    }
    if (scissorDirty_) {
        [encoder_ setScissorRect:{ scX, scY, scW, scH }];
        scissorDirty_ = false;
    }
    if (!frontFaceBound_) {
        [encoder_ setFrontFacingWinding:MTLWindingClockwise];
        frontFaceBound_ = true;
    }
    if (cullDirty_) {
        switch (state_.renderStates[D3DM_RS_CULLMODE]) {
            case D3DM_CULL_NONE: [encoder_ setCullMode:MTLCullModeNone];  break;
            case D3DM_CULL_CW:   [encoder_ setCullMode:MTLCullModeFront]; break;
            default:             [encoder_ setCullMode:MTLCullModeBack];  break;
        }
        cullDirty_ = false;
    }
    if (fillDirty_) {
        [encoder_ setTriangleFillMode:(state_.renderStates[D3DM_RS_FILLMODE] == D3DM_FILL_WIREFRAME)
                                          ? MTLTriangleFillModeLines : MTLTriangleFillModeFill];
        fillDirty_ = false;
    }
    if (slopeDirty_) {
        float slopeScale = u32ToFloat(state_.renderStates[D3DM_RS_SLOPESCALEDEPTHBIAS]);
        [encoder_ setDepthBias:0.0f slopeScale:slopeScale clamp:0.0f];
        slopeDirty_ = false;
    }
    if (blendColorDirty_) {
        D3DCOLOR_T bf = state_.renderStates[D3DM_RS_BLENDFACTOR];
        [encoder_ setBlendColorRed:float((bf >> 16) & 0xFFu) / 255.0f
                             green:float((bf >>  8) & 0xFFu) / 255.0f
                              blue:float((bf >>  0) & 0xFFu) / 255.0f
                             alpha:float((bf >> 24) & 0xFFu) / 255.0f];
        blendColorDirty_ = false;
    }
    if (stencilRefDirty_) {
        [encoder_ setStencilReferenceValue:(state_.renderStates[D3DM_RS_STENCILREF] & 0xFFu)];
        stencilRefDirty_ = false;
    }

    /* Bind vertex buffers */
    if (useUpStream0 && upVb) {
        [encoder_ setVertexBuffer:upVb offset:upVbOffset atIndex:0];
        boundVb_[0] = upVb;
        boundVbOff_[0] = upVbOffset;
        streamDirtyMask_ &= ~uint16_t(1u);
    } else if (state_.streams[0].vb && state_.streams[0].vb->activeBuffer()) {
        state_.streams[0].vb->markUsed(currentCmdSeq_);
        id<MTLBuffer> buf = state_.streams[0].vb->activeBuffer();
        size_t off = state_.streams[0].vb->activeOffset() + state_.streams[0].offset;
        if (buf != boundVb_[0] || off != boundVbOff_[0]) {
            [encoder_ setVertexBuffer:buf offset:off atIndex:0];
            boundVb_[0] = buf;
            boundVbOff_[0] = off;
        }
        streamDirtyMask_ &= ~uint16_t(1u);
    }
    if (streamDirtyMask_ & ~uint16_t(1u)) {
        for (uint32_t i = 1; i < 16; i++) {
            if (state_.streams[i].vb && state_.streams[i].vb->activeBuffer()) {
                state_.streams[i].vb->markUsed(currentCmdSeq_);
                id<MTLBuffer> buf = state_.streams[i].vb->activeBuffer();
                size_t off = state_.streams[i].vb->activeOffset() + state_.streams[i].offset;
                if (buf != boundVb_[i] || off != boundVbOff_[i]) {
                    [encoder_ setVertexBuffer:buf offset:off atIndex:i];
                    boundVb_[i] = buf;
                    boundVbOff_[i] = off;
                }
            }
        }
        streamDirtyMask_ &= uint16_t(1u);
    } else {
        for (uint32_t i = 1; i < 16; i++) {
            if (state_.streams[i].vb) state_.streams[i].vb->markUsed(currentCmdSeq_);
        }
    }
    if (!zeroAttrBound_) {
        [encoder_ setVertexBuffer:pipelineCache_->zeroAttrBuffer() offset:0 atIndex:kVsBufZeroAttr];
        zeroAttrBound_ = true;
    }

    /* Shared state structs for VS & PS */
    auto buildFog = [&]() -> MtlFogCB {
        MtlFogCB fog = {};
        fog.enable = state_.renderStates[D3DM_RS_FOGENABLE] ? 1u : 0u;
        fog.mode   = state_.renderStates[D3DM_RS_FOGTABLEMODE];
        fog.useZ   = (fog.mode != 0) ? 1u : 0u;
        D3DCOLOR_T fc = state_.renderStates[D3DM_RS_FOGCOLOR];
        fog.color[0] = float((fc >> 16) & 0xFFu) / 255.0f;
        fog.color[1] = float((fc >>  8) & 0xFFu) / 255.0f;
        fog.color[2] = float((fc >>  0) & 0xFFu) / 255.0f;
        float fs = u32ToFloat(state_.renderStates[D3DM_RS_FOGSTART]);
        float fe = u32ToFloat(state_.renderStates[D3DM_RS_FOGEND]);
        fog.distEnd   = fe;
        fog.distScale = (fe != fs) ? (1.0f / (fe - fs)) : 0.0f;
        fog.density   = u32ToFloat(state_.renderStates[D3DM_RS_FOGDENSITY]);
        return fog;
    };

    auto buildPointArgs = [&]() -> MtlPointArgsCB {
        MtlPointArgsCB pa = {};
        pa.isPointSprite = state_.renderStates[D3DM_RS_POINTSPRITEENABLE] ? 1u : 0u;
        pa.pointSize     = u32ToFloat(state_.renderStates[D3DM_RS_POINTSIZE]);
        pa.pointSizeMin  = u32ToFloat(state_.renderStates[D3DM_RS_POINTSIZE_MIN]);
        pa.pointSizeMax  = u32ToFloat(state_.renderStates[D3DM_RS_POINTSIZE_MAX]);
        if (pa.pointSizeMax < pa.pointSizeMin) pa.pointSizeMax = std::max(pa.pointSizeMin, 64.0f);
        return pa;
    };

    /* Bind VS CBVs */
    uint32_t vsCbv = vsParsed->defaultCompiled.usedCbvMask;
    if (vsCbv & (1u << 0)) {
        uint16_t needVecs = vsParsed->defaultCompiled.constFloatVecCount;
        if (needVecs == 0 || needVecs > 256u) needVecs = 256u;
        if (boundVsConstFVersion_ != vsConstFVersion_ || boundVsConstFVecs_ < needVecs) {
            [encoder_ setVertexBytes:state_.vsConstF length:size_t(needVecs) * 16u atIndex:kVsBufConstFloat];
            boundVsConstFVersion_ = vsConstFVersion_;
            boundVsConstFVecs_ = needVecs;
        }
    }
    if ((vsCbv & (1u << 1)) && vsConstIDirty_) {
        [encoder_ setVertexBytes:state_.vsConstI length:sizeof(state_.vsConstI) atIndex:kVsBufConstInt];
        vsConstIDirty_ = false;
    }
    if ((vsCbv & (1u << 2)) && vsConstBDirty_) {
        [encoder_ setVertexBytes:state_.vsConstB length:sizeof(state_.vsConstB) atIndex:kVsBufConstBool];
        vsConstBDirty_ = false;
    }
    if ((vsCbv & (1u << 4)) && vsFogDirty_) {
        MtlFogCB fog = buildFog();
        [encoder_ setVertexBytes:&fog length:sizeof(fog) atIndex:kVsBufFog];
        vsFogDirty_ = false;
    }
    if (vsCbv & (1u << 5)) {
        MtlClipPlanesCB cp = {};
        uint32_t mask = state_.renderStates[D3DM_RS_CLIPPLANEENABLE];
        uint32_t cnt = 0;
        for (uint32_t i = 0; i < 6; i++) {
            if (mask & (1u << i)) {
                std::memcpy(cp.planes[cnt++], state_.clipPlanes[i], 16);
            }
        }
        cp.count = cnt;
        [encoder_ setVertexBytes:&cp length:sizeof(cp) atIndex:kVsBufClipPlanes];
    }
    if (vsCbv & (1u << 6)) {
        MtlPointArgsCB pa = buildPointArgs();
        [encoder_ setVertexBytes:&pa length:sizeof(pa) atIndex:kVsBufPointArgs];
    }
    if (vsCbv & (1u << 9)) {
        float depthBias = u32ToFloat(state_.renderStates[D3DM_RS_DEPTHBIAS]);
        float zRange = state_.viewport.MaxZ - state_.viewport.MinZ;
        float clipBias = (zRange > 0.0f) ? (depthBias / zRange) : 0.0f;
        MtlPosFixupCB pf = { 1.0f / float(vpW), -1.0f / float(vpH), 0.0f, clipBias };
        [encoder_ setVertexBytes:&pf length:sizeof(pf) atIndex:kVsBufPosFixup];
    }

    /* Bind VS vertex textures if used (slots 16..19 -> Metal texture 0..3) */
    uint32_t vsSampMask = vsParsed->defaultCompiled.usedSamplerMask;
    if (vsSampMask != 0 || (vsCbv & (1u << 7))) {
        MtlSamplerStateEntry vsEntries[16] = {};
        for (uint32_t s = 0; s < 4; s++) {
            if (!(vsSampMask & (1u << s))) continue;
            uint32_t slot = 16u + s;
            MtlSamplerTexKind expectedKind = vsParsed->defaultCompiled.samplerKind[s];
            MtlTextureCommon *tc = getTextureCommon(state_.textures[slot]);
            bool valid = tc && tc->baseTexture != nil &&
                ((expectedKind == MtlSamplerTexKind::Tex2D   && tc->mtlType == MTLTextureType2D) ||
                 (expectedKind == MtlSamplerTexKind::TexCube && tc->mtlType == MTLTextureTypeCube) ||
                 (expectedKind == MtlSamplerTexKind::Tex3D   && tc->mtlType == MTLTextureType3D));
            if (!valid) {
                vsEntries[s].textureType = uint32_t(expectedKind);
                vsEntries[s].isNull = 1;
                vsEntries[s].drefScale = 1.0f;
                [encoder_ setVertexTexture:pipelineCache_->nullTexture(expectedKind) atIndex:s];
                [encoder_ setVertexSamplerState:pipelineCache_->defaultSampler() atIndex:s];
            } else {
                bool srgb = (state_.samplerStates[slot][D3DM_SAMP_SRGBTEXTURE] != 0);
                vsEntries[s].textureType = uint32_t(expectedKind);
                vsEntries[s].useDepthCompare = (tc->fmtInfo.isDepth && !tc->fmtInfo.isRawDepthFetch) ? 1u : 0u;
                vsEntries[s].drefClamp = 1;
                vsEntries[s].drefScale = 1.0f;
                MtlSamplerKey sk = {};
                sk.addrU = uint8_t(state_.samplerStates[slot][D3DM_SAMP_ADDRESSU]);
                sk.addrV = uint8_t(state_.samplerStates[slot][D3DM_SAMP_ADDRESSV]);
                sk.addrW = uint8_t(state_.samplerStates[slot][D3DM_SAMP_ADDRESSW]);
                sk.magFilter = uint8_t(state_.samplerStates[slot][D3DM_SAMP_MAGFILTER]);
                sk.minFilter = uint8_t(state_.samplerStates[slot][D3DM_SAMP_MINFILTER]);
                sk.mipFilter = uint8_t(state_.samplerStates[slot][D3DM_SAMP_MIPFILTER]);
                sk.maxAniso = uint8_t(state_.samplerStates[slot][D3DM_SAMP_MAXANISOTROPY]);
                sk.maxMipLevel = uint8_t(state_.samplerStates[slot][D3DM_SAMP_MAXMIPLEVEL]);
                sk.depthCompare = (tc->fmtInfo.isDepth && !tc->fmtInfo.isRawDepthFetch) ? 1u : 0u;
                sk.borderColor = state_.samplerStates[slot][D3DM_SAMP_BORDERCOLOR];
                [encoder_ setVertexTexture:tc->getSampleView(srgb) atIndex:s];
                [encoder_ setVertexSamplerState:pipelineCache_->getOrCreateSamplerState(sk) atIndex:s];
            }
        }
        if (vsCbv & (1u << 7)) {
            [encoder_ setVertexBytes:vsEntries length:sizeof(vsEntries) atIndex:kVsBufSamplerState];
        }
    }

    /* Bind PS CBVs, textures & samplers */
    if (psCompiled) {
        uint32_t psCbv = psCompiled->usedCbvMask;
        if (psCbv & (1u << 0)) {
            uint16_t needVecs = psCompiled->constFloatVecCount;
            if (needVecs == 0 || needVecs > 224u) needVecs = 224u;
            if (boundPsConstFVersion_ != psConstFVersion_ || boundPsConstFVecs_ < needVecs) {
                [encoder_ setFragmentBytes:state_.psConstF length:size_t(needVecs) * 16u atIndex:kPsBufConstFloat];
                boundPsConstFVersion_ = psConstFVersion_;
                boundPsConstFVecs_ = needVecs;
            }
        }
        if ((psCbv & (1u << 1)) && psConstIDirty_) {
            [encoder_ setFragmentBytes:state_.psConstI length:sizeof(state_.psConstI) atIndex:kPsBufConstInt];
            psConstIDirty_ = false;
        }
        if ((psCbv & (1u << 2)) && psConstBDirty_) {
            [encoder_ setFragmentBytes:state_.psConstB length:sizeof(state_.psConstB) atIndex:kPsBufConstBool];
            psConstBDirty_ = false;
        }
        if (psCbv & (1u << 3)) {
            MtlAlphaTestCB at = {};
            if (state_.renderStates[D3DM_RS_ALPHATESTENABLE]) {
                uint32_t fn = state_.renderStates[D3DM_RS_ALPHAFUNC];
                at.compareOp = (fn >= 1 && fn <= 8) ? (fn - 1u) : 7u;
            } else {
                at.compareOp = 7u; /* Always */
            }
            bool rt0Float = renderTargets_[0] && renderTargets_[0]->formatInfo().isFloat;
            at.precision = rt0Float ? 9u : 0u;
            at.alphaRef  = state_.renderStates[D3DM_RS_ALPHAREF] & 0xFFu;
            [encoder_ setFragmentBytes:&at length:sizeof(at) atIndex:kPsBufAlphaTest];
        }
        if ((psCbv & (1u << 4)) && psFogDirty_) {
            MtlFogCB fog = buildFog();
            [encoder_ setFragmentBytes:&fog length:sizeof(fog) atIndex:kPsBufFog];
            psFogDirty_ = false;
        }
        if (psCbv & (1u << 6)) {
            MtlPointArgsCB pa = buildPointArgs();
            [encoder_ setFragmentBytes:&pa length:sizeof(pa) atIndex:kPsBufPointArgs];
        }
        if ((psCbv & (1u << 8)) && psTexStageDirty_) {
            MtlTexStageEntry tse[8] = {};
            for (uint32_t s = 0; s < 8; s++) {
                tse[s].bumpMat0[0] = u32ToFloat(state_.texStageStates[s][7]);
                tse[s].bumpMat0[1] = u32ToFloat(state_.texStageStates[s][8]);
                tse[s].bumpMat1[0] = u32ToFloat(state_.texStageStates[s][9]);
                tse[s].bumpMat1[1] = u32ToFloat(state_.texStageStates[s][10]);
                tse[s].bumpScale   = u32ToFloat(state_.texStageStates[s][22]);
                tse[s].bumpOffset  = u32ToFloat(state_.texStageStates[s][23]);
            }
            [encoder_ setFragmentBytes:tse length:sizeof(tse) atIndex:kPsBufTexStage];
            psTexStageDirty_ = false;
        }

        uint32_t psSampMask = psCompiled->usedSamplerMask;
        bool needSampCbv = (psCbv & (1u << 7)) &&
            (psSampMask != boundPsSampCbvMask_ || ((psTexDirtyMask_ | psSampDirtyMask_) & psSampMask) != 0);
        uint32_t checkMask = needSampCbv ? psSampMask : (psSampMask & (psTexDirtyMask_ | psSampDirtyMask_));
        if (checkMask != 0) {
            MtlSamplerStateEntry entries[16];
            if (needSampCbv) std::memset(entries, 0, sizeof(entries));
            while (checkMask) {
                uint32_t s = uint32_t(__builtin_ctz(checkMask));
                checkMask &= (checkMask - 1u);
                MtlSamplerTexKind expectedKind = psCompiled->samplerKind[s];
                MtlTextureCommon *tc = getTextureCommon(state_.textures[s]);
                bool valid = tc && tc->baseTexture != nil &&
                    ((expectedKind == MtlSamplerTexKind::Tex2D   && tc->mtlType == MTLTextureType2D) ||
                     (expectedKind == MtlSamplerTexKind::TexCube && tc->mtlType == MTLTextureTypeCube) ||
                     (expectedKind == MtlSamplerTexKind::Tex3D   && tc->mtlType == MTLTextureType3D));
                if (!valid) {
                    if (needSampCbv) {
                        entries[s].textureType = uint32_t(expectedKind);
                        entries[s].isNull = 1;
                        entries[s].drefScale = 1.0f;
                    }
                    id<MTLTexture> nt = pipelineCache_->nullTexture(expectedKind);
                    id<MTLSamplerState> ds = pipelineCache_->defaultSampler();
                    if (nt != boundFragTex_[s]) {
                        [encoder_ setFragmentTexture:nt atIndex:s];
                        boundFragTex_[s] = nt;
                    }
                    if (ds != boundFragSmp_[s]) {
                        [encoder_ setFragmentSamplerState:ds atIndex:s];
                        boundFragSmp_[s] = ds;
                    }
                } else {
                    if (needSampCbv) {
                        entries[s].textureType     = uint32_t(expectedKind);
                        entries[s].useDepthCompare = (tc->fmtInfo.isDepth && !tc->fmtInfo.isRawDepthFetch) ? 1u : 0u;
                        entries[s].useProjection   = (s < 8 && (state_.texStageStates[s][24] & 256u) != 0) ? 1u : 0u;
                        entries[s].isNull          = 0;
                        entries[s].useGather       = ((state_.fetch4Mask & (1u << s)) != 0) ? 1u : 0u;
                        entries[s].drefClamp       = 1;
                        entries[s].drefScale       = 1.0f;
                    }

                    if (psTexDirtyMask_ & (1u << s)) {
                        bool srgb = (state_.samplerStates[s][D3DM_SAMP_SRGBTEXTURE] != 0);
                        id<MTLTexture> texView = tc->getSampleView(srgb);
                        if (texView != boundFragTex_[s]) {
                            [encoder_ setFragmentTexture:texView atIndex:s];
                            boundFragTex_[s] = texView;
                        }
                    }
                    if ((psSampDirtyMask_ | psTexDirtyMask_) & (1u << s)) {
                        MtlSamplerKey sk = {};
                        sk.addrU       = uint8_t(state_.samplerStates[s][D3DM_SAMP_ADDRESSU]);
                        sk.addrV       = uint8_t(state_.samplerStates[s][D3DM_SAMP_ADDRESSV]);
                        sk.addrW       = uint8_t(state_.samplerStates[s][D3DM_SAMP_ADDRESSW]);
                        sk.magFilter   = uint8_t(state_.samplerStates[s][D3DM_SAMP_MAGFILTER]);
                        sk.minFilter   = uint8_t(state_.samplerStates[s][D3DM_SAMP_MINFILTER]);
                        sk.mipFilter   = uint8_t(state_.samplerStates[s][D3DM_SAMP_MIPFILTER]);
                        sk.maxAniso    = uint8_t(state_.samplerStates[s][D3DM_SAMP_MAXANISOTROPY]);
                        uint32_t maxMip = std::max(state_.samplerStates[s][D3DM_SAMP_MAXMIPLEVEL], tc->lod);
                        sk.maxMipLevel = uint8_t(std::min(maxMip, 255u));
                        sk.depthCompare = (tc->fmtInfo.isDepth && !tc->fmtInfo.isRawDepthFetch) ? 1u : 0u;
                        sk.borderColor = state_.samplerStates[s][D3DM_SAMP_BORDERCOLOR];
                        id<MTLSamplerState> smpObj = pipelineCache_->getOrCreateSamplerState(sk);
                        if (smpObj != boundFragSmp_[s]) {
                            [encoder_ setFragmentSamplerState:smpObj atIndex:s];
                            boundFragSmp_[s] = smpObj;
                        }
                    }
                }
            }
            psTexDirtyMask_ &= ~uint16_t(psSampMask);
            psSampDirtyMask_ &= ~uint16_t(psSampMask);
            if (needSampCbv) {
                [encoder_ setFragmentBytes:entries length:sizeof(entries) atIndex:kPsBufSamplerState];
                boundPsSampCbvMask_ = psSampMask;
            }
        }
    }

    return true;
}

MTL_HRESULT MtlDirect3DDevice9::DrawPrimitive(uint32_t PrimitiveType, uint32_t StartVertex, uint32_t PrimitiveCount) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (PrimitiveCount == 0) return D3D_S_OK;
    if (encoder_ == nil && PrimitiveCount == 2u && state_.streams[0].vb && state_.streams[0].stride >= 8u) {
        id<MTLBuffer> buf = state_.streams[0].vb->activeBuffer();
        uint32_t vCount = primCountToVertexCount(PrimitiveType, PrimitiveCount);
        if (buf != nil && vCount >= 4u) {
            size_t bufLen = size_t([buf length]);
            size_t vbOff = state_.streams[0].vb->activeOffset() + size_t(state_.streams[0].offset);
            size_t startOff = vbOff + size_t(StartVertex) * size_t(state_.streams[0].stride);
            size_t needBytes = size_t(vCount) * size_t(state_.streams[0].stride);
            if (startOff <= bufLen && needBytes <= bufLen - startOff) {
                tryMarkFullScreenDontCare(PrimitiveType, PrimitiveCount,
                                          static_cast<const uint8_t *>([buf contents]) + startOff,
                                          state_.streams[0].stride, vCount);
            }
        }
    }
    if (!prepareDraw(PrimitiveType, false, nil, 0, 0, nullptr)) return D3D_S_OK;

    if (PrimitiveType == D3DM_PT_TRIANGLEFAN) {
        size_t idxCount = size_t(PrimitiveCount) * 3u;
        auto a = ringBuffer_->alloc(idxCount * sizeof(uint32_t), 16);
        uint32_t *idx = static_cast<uint32_t *>(a.cpuPtr);
        for (uint32_t i = 0; i < PrimitiveCount; i++) {
            idx[i * 3 + 0] = 0;
            idx[i * 3 + 1] = i + 1;
            idx[i * 3 + 2] = i + 2;
        }
        [encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                             indexCount:idxCount
                              indexType:MTLIndexTypeUInt32
                            indexBuffer:a.buffer
                      indexBufferOffset:a.offset
                          instanceCount:1
                             baseVertex:NSInteger(StartVertex)
                           baseInstance:0];
        return D3D_S_OK;
    }

    uint32_t vCount = primCountToVertexCount(PrimitiveType, PrimitiveCount);
    if (vCount == 0) return D3D_S_OK;
    [encoder_ drawPrimitives:mapD3dPrimType(PrimitiveType) vertexStart:StartVertex vertexCount:vCount];
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::DrawIndexedPrimitive(uint32_t PrimitiveType, int32_t BaseVertexIndex,
                                                     uint32_t MinVertexIndex, uint32_t NumVertices,
                                                     uint32_t StartIndex, uint32_t PrimCount) {
    (void)MinVertexIndex; (void)NumVertices;
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (PrimCount == 0 || !state_.ib || !state_.ib->activeBuffer()) return D3D_S_OK;
    if (encoder_ == nil && PrimCount == 2u && state_.streams[0].vb && state_.streams[0].stride >= 8u) {
        tryMarkFullScreenDontCareIndexed(PrimitiveType, BaseVertexIndex, StartIndex, PrimCount,
                                         state_.streams[0].vb, state_.streams[0].offset,
                                         state_.streams[0].stride, state_.ib);
    }
    if (!prepareDraw(PrimitiveType, false, nil, 0, 0, nullptr)) return D3D_S_OK;
    state_.ib->markUsed(currentCmdSeq_);

    uint32_t instanceCount = 1;
    if ((state_.streams[0].freq & D3DM_STREAMSOURCE_INDEXEDDATA) != 0) {
        instanceCount = std::max(1u, state_.streams[0].freq & 0x3FFFFFFFu);
    }

    if (PrimitiveType == D3DM_PT_TRIANGLEFAN) {
        size_t outCount = size_t(PrimCount) * 3u;
        auto a = ringBuffer_->alloc(outCount * sizeof(uint32_t), 16);
        uint32_t *dst = static_cast<uint32_t *>(a.cpuPtr);
        const uint8_t *srcBase = static_cast<const uint8_t *>([state_.ib->activeBuffer() contents]) + state_.ib->activeOffset();
        if (state_.ib->indexStride() == 2) {
            const uint16_t *s16 = reinterpret_cast<const uint16_t *>(srcBase) + StartIndex;
            for (uint32_t i = 0; i < PrimCount; i++) {
                dst[i * 3 + 0] = s16[0];
                dst[i * 3 + 1] = s16[i + 1];
                dst[i * 3 + 2] = s16[i + 2];
            }
        } else {
            const uint32_t *s32 = reinterpret_cast<const uint32_t *>(srcBase) + StartIndex;
            for (uint32_t i = 0; i < PrimCount; i++) {
                dst[i * 3 + 0] = s32[0];
                dst[i * 3 + 1] = s32[i + 1];
                dst[i * 3 + 2] = s32[i + 2];
            }
        }
        [encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                             indexCount:outCount
                              indexType:MTLIndexTypeUInt32
                            indexBuffer:a.buffer
                      indexBufferOffset:a.offset
                          instanceCount:instanceCount
                             baseVertex:NSInteger(BaseVertexIndex)
                           baseInstance:0];
        return D3D_S_OK;
    }

    uint32_t idxCount = primCountToVertexCount(PrimitiveType, PrimCount);
    if (idxCount == 0) return D3D_S_OK;

    size_t idxOffset = state_.ib->activeOffset() + size_t(StartIndex) * state_.ib->indexStride();
    [encoder_ drawIndexedPrimitives:mapD3dPrimType(PrimitiveType)
                         indexCount:idxCount
                          indexType:state_.ib->indexType()
                        indexBuffer:state_.ib->activeBuffer()
                  indexBufferOffset:idxOffset
                      instanceCount:instanceCount
                         baseVertex:NSInteger(BaseVertexIndex)
                       baseInstance:0];
    return D3D_S_OK;
}

bool MtlDirect3DDevice9::canUseFullScreenDontCareBase(uint32_t primType, uint32_t primCount, uint32_t minStride,
                                                      int &outPosOff, bool &outIsClipSpace, uint32_t &outPosDeclType,
                                                      uint32_t &outRtW, uint32_t &outRtH) const {
    if (encoder_ != nil || primCount != 2u || minStride < 8u || !state_.vdecl) return false;
    if (primType != D3DM_PT_TRIANGLELIST && primType != D3DM_PT_TRIANGLESTRIP && primType != D3DM_PT_TRIANGLEFAN) return false;
    if (!renderTargets_[0] || renderTargets_[0]->isNull() || renderTargets_[0]->hasPendingColorClear() || renderTargets_[1]) {
        return false;
    }

    bool blendOverwrite = !state_.renderStates[D3DM_RS_ALPHABLENDENABLE] ||
        (!state_.renderStates[D3DM_RS_SEPARATEALPHABLENDENABLE] &&
         state_.renderStates[D3DM_RS_SRCBLEND] == D3DM_BLEND_ONE &&
         state_.renderStates[D3DM_RS_DESTBLEND] == D3DM_BLEND_ZERO &&
         state_.renderStates[D3DM_RS_BLENDOP] == D3DM_BLENDOP_ADD);
    bool depthIgnore = !state_.renderStates[D3DM_RS_ZENABLE] ||
        state_.renderStates[D3DM_RS_ZFUNC] == D3DM_CMP_ALWAYS;
    if (!blendOverwrite || state_.renderStates[D3DM_RS_ALPHATESTENABLE] || !depthIgnore) {
        return false;
    }
    if ((state_.renderStates[D3DM_RS_COLORWRITEENABLE] & 0xFu) != 0xFu) return false;

    uint32_t rtW = renderTargets_[0]->desc().Width;
    uint32_t rtH = renderTargets_[0]->desc().Height;
    if (rtW < 64u || rtH < 64u) return false;
    if (state_.viewport.X != 0 || state_.viewport.Y != 0 ||
        state_.viewport.Width < rtW || state_.viewport.Height < rtH) {
        return false;
    }
    if (state_.renderStates[D3DM_RS_SCISSORTESTENABLE]) {
        if (state_.scissorRect.left > 0 || state_.scissorRect.top > 0 ||
            state_.scissorRect.right < int32_t(rtW) || state_.scissorRect.bottom < int32_t(rtH)) {
            return false;
        }
    }

    int posOff = -1;
    bool isClipSpace = false;
    uint32_t posDeclType = 0;
    for (const auto &el : state_.vdecl->elements()) {
        if (el.Stream == 0 && el.Usage == 9 /* POSITIONT */ && el.UsageIndex == 0 &&
            (el.Type == D3DM_DECLTYPE_FLOAT2 || el.Type == D3DM_DECLTYPE_FLOAT3 || el.Type == D3DM_DECLTYPE_FLOAT4)) {
            posOff = int(el.Offset);
            posDeclType = el.Type;
            break;
        }
    }
    if (posOff < 0 && state_.vs && state_.vs->parsed() && state_.vs->parsed()->vsClipSpacePos &&
        (!state_.ps || !state_.ps->parsed() || !state_.ps->parsed()->psHasTexkill) &&
        state_.renderStates[D3DM_RS_CULLMODE] == 1 /* D3DCULL_NONE */ &&
        !state_.renderStates[D3DM_RS_STENCILENABLE]) {
        for (const auto &el : state_.vdecl->elements()) {
            if (el.Stream == 0 && el.Usage == 0 /* POSITION */ && el.UsageIndex == 0 &&
                (el.Type == D3DM_DECLTYPE_FLOAT2 || el.Type == D3DM_DECLTYPE_FLOAT3 || el.Type == D3DM_DECLTYPE_FLOAT4)) {
                posOff = int(el.Offset);
                posDeclType = el.Type;
                isClipSpace = true;
                break;
            }
        }
    }
    uint32_t minBytes = (isClipSpace && posDeclType == D3DM_DECLTYPE_FLOAT4) ? 16u : 8u;
    if (posOff < 0 || uint32_t(posOff + minBytes) > minStride) return false;

    outPosOff = posOff;
    outIsClipSpace = isClipSpace;
    outPosDeclType = posDeclType;
    outRtW = rtW;
    outRtH = rtH;
    return true;
}

static bool evalFullScreenQuadVertex(const float *xyzw, bool isClipSpace, uint32_t posDeclType,
                                     uint32_t rtW, uint32_t rtH, uint32_t &cornerMask) {
    if (isClipSpace) {
        float w = (posDeclType == D3DM_DECLTYPE_FLOAT4) ? xyzw[3] : 1.0f;
        if (w <= 1e-4f) return false;
        float x = xyzw[0] / w;
        float y = xyzw[1] / w;
        uint32_t bit = 0;
        if (x <= -0.999f && y <= -0.999f) bit = 1u;
        else if (x >= 0.999f && y <= -0.999f) bit = 2u;
        else if (x <= -0.999f && y >= 0.999f) bit = 4u;
        else if (x >= 0.999f && y >= 0.999f) bit = 8u;
        else return false;
        cornerMask |= bit;
        return true;
    }

    float x = xyzw[0];
    float y = xyzw[1];
    float maxW = float(rtW) - 0.5f;
    float maxH = float(rtH) - 0.5f;
    uint32_t bit = 0;
    if (x <= 0.5f && y <= 0.5f) bit = 1u;
    else if (x >= maxW && y <= 0.5f) bit = 2u;
    else if (x <= 0.5f && y >= maxH) bit = 4u;
    else if (x >= maxW && y >= maxH) bit = 8u;
    else return false;
    cornerMask |= bit;
    return true;
}

void MtlDirect3DDevice9::tryMarkFullScreenDontCare(uint32_t primType, uint32_t primCount,
                                                    const void *vbData, uint32_t stride, uint32_t vertCount) {
    if (!vbData || vertCount < 4u) return;
    int posOff = -1;
    bool isClipSpace = false;
    uint32_t posDeclType = 0, rtW = 0, rtH = 0;
    if (!canUseFullScreenDontCareBase(primType, primCount, stride, posOff, isClipSpace, posDeclType, rtW, rtH)) {
        return;
    }

    const auto *base = static_cast<const uint8_t *>(vbData);
    uint32_t cornerMask = 0;
    for (uint32_t v = 0; v < vertCount; v++) {
        const float *xyzw = reinterpret_cast<const float *>(base + size_t(v) * stride + posOff);
        if (!evalFullScreenQuadVertex(xyzw, isClipSpace, posDeclType, rtW, rtH, cornerMask)) return;
    }
    if (cornerMask == 0xFu) {
        pendingColorDontCare_ = true;
    }
}

void MtlDirect3DDevice9::tryMarkFullScreenDontCareIndexed(uint32_t primType, int32_t baseVertexIndex,
                                                           uint32_t startIndex, uint32_t primCount,
                                                           MtlVertexBuffer9 *vb, uint32_t vbByteOffset,
                                                           uint32_t stride, MtlIndexBuffer9 *ib) {
    if (!vb || !ib || !vb->activeBuffer() || !ib->activeBuffer()) return;
    uint32_t idxCount = primCountToVertexCount(primType, primCount);
    if (idxCount < 4u || idxCount > 6u) return;

    int posOff = -1;
    bool isClipSpace = false;
    uint32_t posDeclType = 0, rtW = 0, rtH = 0;
    if (!canUseFullScreenDontCareBase(primType, primCount, stride, posOff, isClipSpace, posDeclType, rtW, rtH)) {
        return;
    }

    id<MTLBuffer> ibBuf = ib->activeBuffer();
    id<MTLBuffer> vbBuf = vb->activeBuffer();
    size_t ibLen = size_t([ibBuf length]);
    size_t vbLen = size_t([vbBuf length]);
    uint32_t iStride = ib->indexStride();
    size_t ibStart = ib->activeOffset() + size_t(startIndex) * iStride;
    size_t ibNeed = size_t(idxCount) * iStride;
    if (ibStart > ibLen || ibNeed > ibLen - ibStart) return;

    const uint8_t *ibPtr = static_cast<const uint8_t *>([ibBuf contents]) + ibStart;
    const uint8_t *vbRaw = static_cast<const uint8_t *>([vbBuf contents]);
    size_t vbBaseOff = vb->activeOffset() + size_t(vbByteOffset);
    if (vbBaseOff > vbLen) return;

    uint32_t cornerMask = 0;
    for (uint32_t i = 0; i < idxCount; i++) {
        uint32_t rawIdx = (iStride == 4u)
            ? reinterpret_cast<const uint32_t *>(ibPtr)[i]
            : uint32_t(reinterpret_cast<const uint16_t *>(ibPtr)[i]);
        int64_t vIdx = int64_t(baseVertexIndex) + int64_t(rawIdx);
        if (vIdx < 0) return;
        size_t vertOff = vbBaseOff + size_t(vIdx) * stride;
        if (vertOff > vbLen || stride > vbLen - vertOff) return;
        const float *xyzw = reinterpret_cast<const float *>(vbRaw + vertOff + posOff);
        if (!evalFullScreenQuadVertex(xyzw, isClipSpace, posDeclType, rtW, rtH, cornerMask)) return;
    }
    if (cornerMask == 0xFu) {
        pendingColorDontCare_ = true;
    }
}

MTL_HRESULT MtlDirect3DDevice9::DrawPrimitiveUP(uint32_t PrimitiveType, uint32_t PrimitiveCount,
                                                const void *pVertexStreamZeroData, uint32_t VertexStreamZeroStride) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (PrimitiveCount == 0 || !pVertexStreamZeroData || VertexStreamZeroStride == 0) return D3D_S_OK;

    uint32_t vCount = primCountToVertexCount(PrimitiveType, PrimitiveCount);
    if (vCount == 0) return D3D_S_OK;

    tryMarkFullScreenDontCare(PrimitiveType, PrimitiveCount, pVertexStreamZeroData, VertexStreamZeroStride, vCount);

    size_t vbBytes = size_t(vCount) * VertexStreamZeroStride;
    auto vbAlloc = ringBuffer_->alloc(vbBytes, 16);
    std::memcpy(vbAlloc.cpuPtr, pVertexStreamZeroData, vbBytes);

    if (prepareDraw(PrimitiveType, true, vbAlloc.buffer, vbAlloc.offset, VertexStreamZeroStride, nullptr)) {
        if (PrimitiveType == D3DM_PT_TRIANGLEFAN) {
            size_t idxCount = size_t(PrimitiveCount) * 3u;
            auto ia = ringBuffer_->alloc(idxCount * sizeof(uint32_t), 16);
            uint32_t *idx = static_cast<uint32_t *>(ia.cpuPtr);
            for (uint32_t i = 0; i < PrimitiveCount; i++) {
                idx[i * 3 + 0] = 0;
                idx[i * 3 + 1] = i + 1;
                idx[i * 3 + 2] = i + 2;
            }
            [encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                 indexCount:idxCount
                                  indexType:MTLIndexTypeUInt32
                                indexBuffer:ia.buffer
                          indexBufferOffset:ia.offset];
        } else {
            [encoder_ drawPrimitives:mapD3dPrimType(PrimitiveType) vertexStart:0 vertexCount:vCount];
        }
    }

    /* D3D9 spec: DrawPrimitiveUP clears stream 0 */
    SetStreamSource(0, nullptr, 0, 0);
    return D3D_S_OK;
}

MTL_HRESULT MtlDirect3DDevice9::DrawIndexedPrimitiveUP(uint32_t PrimitiveType, uint32_t MinVertexIndex,
                                                       uint32_t NumVertices, uint32_t PrimitiveCount,
                                                       const void *pIndexData, uint32_t IndexDataFormat,
                                                       const void *pVertexStreamZeroData, uint32_t VertexStreamZeroStride) {
    std::lock_guard<MtlFastRecursiveLock> lk(mu_);
    if (PrimitiveCount == 0 || !pIndexData || !pVertexStreamZeroData || VertexStreamZeroStride == 0) return D3D_S_OK;

    uint32_t idxCount = primCountToVertexCount(PrimitiveType, PrimitiveCount);
    if (idxCount == 0) return D3D_S_OK;

    bool is32 = (IndexDataFormat == D3DM_FMT_INDEX32);
    uint32_t maxVert = MinVertexIndex + NumVertices;
    if (is32) {
        const uint32_t *s = static_cast<const uint32_t *>(pIndexData);
        for (uint32_t i = 0; i < idxCount; i++) maxVert = std::max(maxVert, s[i] + 1u);
    } else {
        const uint16_t *s = static_cast<const uint16_t *>(pIndexData);
        for (uint32_t i = 0; i < idxCount; i++) maxVert = std::max(maxVert, uint32_t(s[i]) + 1u);
    }

    size_t vbBytes = size_t(std::max(maxVert, 1u)) * VertexStreamZeroStride;
    auto vbAlloc = ringBuffer_->alloc(vbBytes, 16);
    std::memcpy(vbAlloc.cpuPtr, pVertexStreamZeroData, vbBytes);

    /* UE3 renders the menu cursor as a fixed 32x32 px DXT5 quad regardless of resolution, making it tiny at 4K.
       Scale the quad proportionally above 720p (override via ME_PC_CURSOR_SCALE). */
    if (PrimitiveType == D3DM_PT_TRIANGLELIST && PrimitiveCount == 2 && NumVertices == 4 &&
        MinVertexIndex == 0 && primarySwapChain_ && renderTargets_[0] &&
        renderTargets_[0] == primarySwapChain_->backBuffer() &&
        presentParams_.BackBufferHeight > 720 && state_.vdecl && VertexStreamZeroStride >= 8) {
        MtlTextureCommon *tex0 = getTextureCommon(state_.textures[0]);
        if (tex0 && tex0->mtlType == MTLTextureType2D && tex0->width == 32 && tex0->height == 32 &&
            tex0->d3dFormat == D3DM_FMT_DXT5) {
            static float envCursorScale = -1.0f;
            if (envCursorScale < 0.0f) {
                const char *e = getenv("ME_PC_CURSOR_SCALE");
                envCursorScale = (e && *e) ? float(atof(e)) : 0.0f;
            }
            float scale = (envCursorScale > 0.0f) ? envCursorScale : (float(presentParams_.BackBufferHeight) / 720.0f);
            if (scale > 1.05f) {
                int posOff = -1;
                for (const auto &el : state_.vdecl->elements()) {
                    if (el.Stream == 0 && (el.Usage == 0 || el.Usage == 9) && el.UsageIndex == 0 &&
                        (el.Type == D3DM_DECLTYPE_FLOAT2 || el.Type == D3DM_DECLTYPE_FLOAT3 || el.Type == D3DM_DECLTYPE_FLOAT4)) {
                        posOff = int(el.Offset);
                        break;
                    }
                }
                if (posOff >= 0 && uint32_t(posOff + 8) <= VertexStreamZeroStride) {
                    auto *base = static_cast<uint8_t *>(vbAlloc.cpuPtr);
                    float minX = 1e30f, minY = 1e30f, maxX = -1e30f, maxY = -1e30f;
                    for (uint32_t v = 0; v < 4; v++) {
                        const float *xy = reinterpret_cast<const float *>(base + v * VertexStreamZeroStride + posOff);
                        minX = std::min(minX, xy[0]); maxX = std::max(maxX, xy[0]);
                        minY = std::min(minY, xy[1]); maxY = std::max(maxY, xy[1]);
                    }
                    float qw = maxX - minX, qh = maxY - minY;
                    if (qw >= 16.0f && qw <= 40.0f && qh >= 16.0f && qh <= 40.0f) {
                        int cx = 0, cy = 0;
                        bool nearCursor = !host_cursor_pos || !host_cursor_pos(&cx, &cy) ||
                                          (std::fabs(float(cx) - minX) <= 48.0f && std::fabs(float(cy) - minY) <= 48.0f);
                        if (nearCursor) {
                            float anchorX = (host_cursor_pos && host_cursor_pos(&cx, &cy) &&
                                             float(cx) >= minX - 4.0f && float(cx) <= maxX + 4.0f) ? float(cx) : minX;
                            float anchorY = (host_cursor_pos && host_cursor_pos(&cx, &cy) &&
                                             float(cy) >= minY - 4.0f && float(cy) <= maxY + 4.0f) ? float(cy) : minY;
                            for (uint32_t v = 0; v < 4; v++) {
                                float *xy = reinterpret_cast<float *>(base + v * VertexStreamZeroStride + posOff);
                                xy[0] = anchorX + (xy[0] - anchorX) * scale;
                                xy[1] = anchorY + (xy[1] - anchorY) * scale;
                            }
                        }
                    }
                }
            }
        }
    }

    tryMarkFullScreenDontCare(PrimitiveType, PrimitiveCount, vbAlloc.cpuPtr, VertexStreamZeroStride, maxVert);

    if (prepareDraw(PrimitiveType, true, vbAlloc.buffer, vbAlloc.offset, VertexStreamZeroStride, nullptr)) {
        if (PrimitiveType == D3DM_PT_TRIANGLEFAN) {
            size_t outCount = size_t(PrimitiveCount) * 3u;
            auto ia = ringBuffer_->alloc(outCount * sizeof(uint32_t), 16);
            uint32_t *dst = static_cast<uint32_t *>(ia.cpuPtr);
            if (is32) {
                const uint32_t *s = static_cast<const uint32_t *>(pIndexData);
                for (uint32_t i = 0; i < PrimitiveCount; i++) {
                    dst[i * 3 + 0] = s[0]; dst[i * 3 + 1] = s[i + 1]; dst[i * 3 + 2] = s[i + 2];
                }
            } else {
                const uint16_t *s = static_cast<const uint16_t *>(pIndexData);
                for (uint32_t i = 0; i < PrimitiveCount; i++) {
                    dst[i * 3 + 0] = s[0]; dst[i * 3 + 1] = s[i + 1]; dst[i * 3 + 2] = s[i + 2];
                }
            }
            [encoder_ drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                 indexCount:outCount
                                  indexType:MTLIndexTypeUInt32
                                indexBuffer:ia.buffer
                          indexBufferOffset:ia.offset];
        } else {
            size_t ibBytes = size_t(idxCount) * (is32 ? 4u : 2u);
            auto ibAlloc = ringBuffer_->alloc(ibBytes, 16);
            std::memcpy(ibAlloc.cpuPtr, pIndexData, ibBytes);
            [encoder_ drawIndexedPrimitives:mapD3dPrimType(PrimitiveType)
                                 indexCount:idxCount
                                  indexType:(is32 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16)
                                indexBuffer:ibAlloc.buffer
                          indexBufferOffset:ibAlloc.offset];
        }
    }

    /* D3D9 spec: DrawIndexedPrimitiveUP clears stream 0 and indices */
    SetStreamSource(0, nullptr, 0, 0);
    SetIndices(nullptr);
    return D3D_S_OK;
}

} // namespace mtl_d3d9

extern "C" void mtl_warm_set_map(const char *name) {
    if (!name || !*name) return;
    std::lock_guard<std::mutex> lk(mtl_d3d9::gWarmDevMu);
    mtl_d3d9::gPendingWarmMap = name;
    if (mtl_d3d9::gWarmDev) {
        mtl_d3d9::gWarmDev->onWarmSetMap(name);
    }
}
