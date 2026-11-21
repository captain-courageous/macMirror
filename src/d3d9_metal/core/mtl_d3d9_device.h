#pragma once

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <SDL3/SDL.h>

#include "d3d9_metal_com.h"
#include "d3d9_metal_enums.h"
#include "mtl_format.h"
#include "mtl_resources.h"
#include "mtl_pipeline_cache.h"
#include "mtl_shader_compiler.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <dispatch/dispatch.h>
#include <memory>
#include <mutex>
#include <os/lock.h>
#include <pthread.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mtl_d3d9 {

/* Lightweight recursive lock backed by os_unfair_lock (~3-5x faster uncontended than std::recursive_mutex). */
class MtlFastRecursiveLock {
public:
    void lock() {
        pthread_t self = pthread_self();
        if (owner_.load(std::memory_order_relaxed) == self) {
            depth_++;
            return;
        }
        os_unfair_lock_lock(&lock_);
        owner_.store(self, std::memory_order_relaxed);
        depth_ = 1;
    }
    void unlock() {
        if (--depth_ == 0) {
            owner_.store(nullptr, std::memory_order_relaxed);
            os_unfair_lock_unlock(&lock_);
        }
    }
private:
    os_unfair_lock lock_ = OS_UNFAIR_LOCK_INIT;
    std::atomic<pthread_t> owner_{nullptr};
    uint32_t depth_ = 0;
};

class MtlDirect3D9;
class MtlDirect3DDevice9;

static constexpr uint32_t kMaxSamplers        = 20; /* 0..15 PS, 16..19 VS (257..260) */
static constexpr uint32_t kMaxTexStages       = 8;
static constexpr uint32_t kMaxTexStageStates  = 33;
static constexpr uint32_t kMaxVisibilitySlots = 8192;
static constexpr uint32_t kInvalidVisSlot     = 0xFFFFFFFFu;

struct MtlStreamState {
    MtlVertexBuffer9 *vb     = nullptr;
    uint32_t          offset = 0;
    uint32_t          stride = 0;
    uint32_t          freq   = 1;
};

struct MtlDeviceState {
    uint32_t renderStates[D3DM_RS_MAX] = {};
    uint32_t samplerStates[kMaxSamplers][D3DM_SAMP_MAX] = {};
    uint32_t texStageStates[kMaxTexStages][kMaxTexStageStates] = {};
    uint32_t fetch4Mask = 0;

    IMtlDirect3DBaseTexture9 *textures[kMaxSamplers] = {};
    MtlVertexShader9         *vs = nullptr;
    MtlPixelShader9          *ps = nullptr;
    MtlVertexDeclaration9    *vdecl = nullptr;
    uint32_t                  fvf = 0;
    MtlStreamState            streams[16] = {};
    MtlIndexBuffer9          *ib = nullptr;

    D3DMtlViewport viewport = {};
    D3DMtlRect     scissorRect = {};
    float          clipPlanes[6][4] = {};

    float    vsConstF[256][4] = {};
    int32_t  vsConstI[16][4] = {};
    uint32_t vsConstB[16] = {};

    float    psConstF[224][4] = {};
    int32_t  psConstI[16][4] = {};
    uint32_t psConstB[16] = {};

    float    transforms[4][16] = {}; /* view, proj, world0..1 (for GetTransform) */

    void resetDefaults(uint32_t bbW, uint32_t bbH);
    void releaseRefs();
};

/* ---- State Block ---- */
class MtlStateBlock9 final : public MtlComObject<MtlStateBlock9, IMtlDirect3DStateBlock9> {
public:
    MtlStateBlock9(MtlDirect3DDevice9 *dev, uint32_t type);
    ~MtlStateBlock9();

    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT Capture() override;
    MTL_HRESULT Apply() override;

    void recordRenderState(uint32_t state, uint32_t val);
    void recordSamplerState(uint32_t slot, uint32_t type, uint32_t val);
    void recordTexStageState(uint32_t stage, uint32_t type, uint32_t val);
    void recordTexture(uint32_t slot, IMtlDirect3DBaseTexture9 *tex);
    void recordVs(MtlVertexShader9 *vs);
    void recordPs(MtlPixelShader9 *ps);
    void recordVdecl(MtlVertexDeclaration9 *decl);
    void recordFvf(uint32_t fvf);
    void recordStream(uint32_t idx, MtlVertexBuffer9 *vb, uint32_t offset, uint32_t stride);
    void recordStreamFreq(uint32_t idx, uint32_t freq);
    void recordIndices(MtlIndexBuffer9 *ib);
    void recordViewport(const D3DMtlViewport &vp);
    void recordScissor(const D3DMtlRect &r);
    void recordClipPlane(uint32_t idx, const float p[4]);
    void recordVsConstF(uint32_t start, const float *data, uint32_t count);
    void recordVsConstI(uint32_t start, const int32_t *data, uint32_t count);
    void recordVsConstB(uint32_t start, const uint32_t *data, uint32_t count);
    void recordPsConstF(uint32_t start, const float *data, uint32_t count);
    void recordPsConstI(uint32_t start, const int32_t *data, uint32_t count);
    void recordPsConstB(uint32_t start, const uint32_t *data, uint32_t count);

private:
    MtlDirect3DDevice9 *device_;
    uint32_t type_;

    bool captureAllPixel_ = false;
    bool captureAllVertex_ = false;

    std::vector<std::pair<uint32_t, uint32_t>> rs_;
    std::vector<std::pair<uint32_t, uint32_t>> ss_;   /* key = (slot << 8) | type */
    std::vector<std::pair<uint32_t, uint32_t>> tss_;  /* key = (stage << 8) | type */
    uint32_t texMask_ = 0;
    IMtlDirect3DBaseTexture9 *textures_[kMaxSamplers] = {};

    bool hasVs_ = false;   MtlVertexShader9 *vs_ = nullptr;
    bool hasPs_ = false;   MtlPixelShader9 *ps_ = nullptr;
    bool hasDecl_ = false; MtlVertexDeclaration9 *vdecl_ = nullptr;
    bool hasFvf_ = false;  uint32_t fvf_ = 0;
    uint32_t streamMask_ = 0;
    uint32_t streamFreqMask_ = 0;
    MtlStreamState streams_[16] = {};
    bool hasIb_ = false;   MtlIndexBuffer9 *ib_ = nullptr;
    bool hasVp_ = false;   D3DMtlViewport vp_ = {};
    bool hasScissor_ = false; D3DMtlRect scissor_ = {};
    uint32_t clipMask_ = 0;
    float clipPlanes_[6][4] = {};

    uint8_t vsConstFMask_[256] = {};
    float   vsConstF_[256][4] = {};
    uint16_t vsConstIMask_ = 0;
    int32_t vsConstI_[16][4] = {};
    uint16_t vsConstBMask_ = 0;
    uint32_t vsConstB_[16] = {};

    uint8_t psConstFMask_[224] = {};
    float   psConstF_[224][4] = {};
    uint16_t psConstIMask_ = 0;
    int32_t psConstI_[16][4] = {};
    uint16_t psConstBMask_ = 0;
    uint32_t psConstB_[16] = {};

    void clearRecorded();
};

/* ---- SwapChain + CAMetalLayer ---- */
class MtlSwapChain9 final : public MtlComObject<MtlSwapChain9, IMtlDirect3DSwapChain9> {
public:
    MtlSwapChain9(MtlDirect3DDevice9 *dev, const D3DMtlPresentParameters &pp);
    ~MtlSwapChain9();

    MTL_HRESULT Present(const D3DMtlRect *pSourceRect, const D3DMtlRect *pDestRect,
                        void *hDestWindowOverride, const void *pDirtyRegion, uint32_t dwFlags) override;
    MTL_HRESULT GetFrontBufferData(IMtlDirect3DSurface9 *pDestSurface) override;
    MTL_HRESULT GetBackBuffer(uint32_t iBackBuffer, uint32_t Type, IMtlDirect3DSurface9 **ppBackBuffer) override;
    MTL_HRESULT GetRasterStatus(D3DMtlRasterStatus *pRasterStatus) override;
    MTL_HRESULT GetDisplayMode(D3DMtlDisplayMode *pMode) override;
    MTL_HRESULT GetDevice(IMtlDirect3DDevice9 **ppDevice) override;
    MTL_HRESULT GetPresentParameters(D3DMtlPresentParameters *pPresentationParameters) override;

    bool init(const D3DMtlPresentParameters &pp);
    bool resize(const D3DMtlPresentParameters &pp);

    MtlSurface9 *backBuffer() const { return backBuffer_; }
    CAMetalLayer *metalLayer() const { return metalLayer_; }
    const D3DMtlPresentParameters &presentParams() const { return pp_; }

private:
    void attachLayerToWindow(SDL_Window *win);
    void detachLayer();
    void updateDrawableSize();

    MtlDirect3DDevice9 *device_;
    D3DMtlPresentParameters pp_ = {};
    SDL_Window   *sdlWindow_ = nullptr;
    SDL_MetalView metalView_ = nullptr;
    CAMetalLayer *metalLayer_ = nil;
    MtlSurface9  *backBuffer_ = nullptr;
};

/* ---- Main Native Metal IDirect3DDevice9 ---- */
class MtlDirect3DDevice9 final : public MtlComObject<MtlDirect3DDevice9, IMtlDirect3DDevice9> {
public:
    MtlDirect3DDevice9(IMtlDirect3D9 *d3d9, id<MTLDevice> mtlDevice,
                       uint32_t adapter, uint32_t devType, void *focusWindow,
                       uint32_t behaviorFlags, const D3DMtlPresentParameters &pp);
    ~MtlDirect3DDevice9();

    bool initialize();

    /* IMtlDirect3DDevice9 */
    MTL_HRESULT TestCooperativeLevel() override { return D3D_S_OK; }
    uint32_t    GetAvailableTextureMem() override;
    MTL_HRESULT EvictManagedResources() override { return D3D_S_OK; }
    MTL_HRESULT GetDirect3D(IMtlDirect3D9 **ppD3D9) override;
    MTL_HRESULT GetDeviceCaps(D3DMtlCaps9 *pCaps) override;
    MTL_HRESULT GetDisplayMode(uint32_t iSwapChain, D3DMtlDisplayMode *pMode) override;
    MTL_HRESULT GetCreationParameters(D3DMtlCreationParameters *pParameters) override;
    MTL_HRESULT SetCursorProperties(uint32_t, uint32_t, IMtlDirect3DSurface9 *) override { return D3D_S_OK; }
    void        SetCursorPosition(int32_t, int32_t, uint32_t) override {}
    uint32_t    ShowCursor(uint32_t) override { return 0; }
    MTL_HRESULT CreateAdditionalSwapChain(D3DMtlPresentParameters *pPP, IMtlDirect3DSwapChain9 **pSwapChain) override;
    MTL_HRESULT GetSwapChain(uint32_t iSwapChain, IMtlDirect3DSwapChain9 **pSwapChain) override;
    uint32_t    GetNumberOfSwapChains() override { return 1; }
    MTL_HRESULT Reset(D3DMtlPresentParameters *pPP) override;
    MTL_HRESULT Present(const D3DMtlRect *pSourceRect, const D3DMtlRect *pDestRect, void *hDestWindowOverride, const void *pDirtyRegion) override;
    MTL_HRESULT GetBackBuffer(uint32_t iSwapChain, uint32_t iBackBuffer, uint32_t Type, IMtlDirect3DSurface9 **ppBackBuffer) override;
    MTL_HRESULT GetRasterStatus(uint32_t iSwapChain, D3DMtlRasterStatus *pRasterStatus) override;
    MTL_HRESULT SetDialogBoxMode(uint32_t) override { return D3D_S_OK; }
    void        SetGammaRamp(uint32_t, uint32_t, const D3DMtlGammaRamp *pRamp) override;
    void        GetGammaRamp(uint32_t, D3DMtlGammaRamp *pRamp) override;
    MTL_HRESULT CreateTexture(uint32_t Width, uint32_t Height, uint32_t Levels, uint32_t Usage, uint32_t Format, uint32_t Pool, IMtlDirect3DTexture9 **ppTexture, void **pSharedHandle) override;
    MTL_HRESULT CreateVolumeTexture(uint32_t Width, uint32_t Height, uint32_t Depth, uint32_t Levels, uint32_t Usage, uint32_t Format, uint32_t Pool, IMtlDirect3DVolumeTexture9 **ppVolumeTexture, void **pSharedHandle) override;
    MTL_HRESULT CreateCubeTexture(uint32_t EdgeLength, uint32_t Levels, uint32_t Usage, uint32_t Format, uint32_t Pool, IMtlDirect3DCubeTexture9 **ppCubeTexture, void **pSharedHandle) override;
    MTL_HRESULT CreateVertexBuffer(uint32_t Length, uint32_t Usage, uint32_t FVF, uint32_t Pool, IMtlDirect3DVertexBuffer9 **ppVertexBuffer, void **pSharedHandle) override;
    MTL_HRESULT CreateIndexBuffer(uint32_t Length, uint32_t Usage, uint32_t Format, uint32_t Pool, IMtlDirect3DIndexBuffer9 **ppIndexBuffer, void **pSharedHandle) override;
    MTL_HRESULT CreateRenderTarget(uint32_t Width, uint32_t Height, uint32_t Format, uint32_t MultiSample, uint32_t MultisampleQuality, uint32_t Lockable, IMtlDirect3DSurface9 **ppSurface, void **pSharedHandle) override;
    MTL_HRESULT CreateDepthStencilSurface(uint32_t Width, uint32_t Height, uint32_t Format, uint32_t MultiSample, uint32_t MultisampleQuality, uint32_t Discard, IMtlDirect3DSurface9 **ppSurface, void **pSharedHandle) override;
    MTL_HRESULT UpdateSurface(IMtlDirect3DSurface9 *pSourceSurface, const D3DMtlRect *pSourceRect, IMtlDirect3DSurface9 *pDestinationSurface, const D3DMtlPoint *pDestPoint) override;
    MTL_HRESULT UpdateTexture(IMtlDirect3DBaseTexture9 *pSourceTexture, IMtlDirect3DBaseTexture9 *pDestinationTexture) override;
    MTL_HRESULT GetRenderTargetData(IMtlDirect3DSurface9 *pRenderTarget, IMtlDirect3DSurface9 *pDestSurface) override;
    MTL_HRESULT GetFrontBufferData(uint32_t iSwapChain, IMtlDirect3DSurface9 *pDestSurface) override;
    MTL_HRESULT StretchRect(IMtlDirect3DSurface9 *pSourceSurface, const D3DMtlRect *pSourceRect, IMtlDirect3DSurface9 *pDestSurface, const D3DMtlRect *pDestRect, uint32_t Filter) override;
    MTL_HRESULT ColorFill(IMtlDirect3DSurface9 *pSurface, const D3DMtlRect *pRect, D3DCOLOR_T color) override;
    MTL_HRESULT CreateOffscreenPlainSurface(uint32_t Width, uint32_t Height, uint32_t Format, uint32_t Pool, IMtlDirect3DSurface9 **ppSurface, void **pSharedHandle) override;
    MTL_HRESULT SetRenderTarget(uint32_t RenderTargetIndex, IMtlDirect3DSurface9 *pRenderTarget) override;
    MTL_HRESULT GetRenderTarget(uint32_t RenderTargetIndex, IMtlDirect3DSurface9 **ppRenderTarget) override;
    MTL_HRESULT SetDepthStencilSurface(IMtlDirect3DSurface9 *pNewZStencil) override;
    MTL_HRESULT GetDepthStencilSurface(IMtlDirect3DSurface9 **ppZStencilSurface) override;
    MTL_HRESULT BeginScene() override;
    MTL_HRESULT EndScene() override;
    MTL_HRESULT Clear(uint32_t Count, const D3DMtlRect *pRects, uint32_t Flags, D3DCOLOR_T Color, float Z, uint32_t Stencil) override;
    MTL_HRESULT SetTransform(uint32_t State, const float *pMatrix) override;
    MTL_HRESULT GetTransform(uint32_t State, float *pMatrix) override;
    MTL_HRESULT MultiplyTransform(uint32_t State, const float *pMatrix) override;
    MTL_HRESULT SetViewport(const D3DMtlViewport *pViewport) override;
    MTL_HRESULT GetViewport(D3DMtlViewport *pViewport) override;
    MTL_HRESULT SetMaterial(const void *) override { return D3D_S_OK; }
    MTL_HRESULT GetMaterial(void *) override { return D3D_S_OK; }
    MTL_HRESULT SetLight(uint32_t, const void *) override { return D3D_S_OK; }
    MTL_HRESULT GetLight(uint32_t, void *) override { return D3D_S_OK; }
    MTL_HRESULT LightEnable(uint32_t, uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetLightEnable(uint32_t, uint32_t *pEnable) override { if (pEnable) *pEnable = 0; return D3D_S_OK; }
    MTL_HRESULT SetClipPlane(uint32_t Index, const float *pPlane) override;
    MTL_HRESULT GetClipPlane(uint32_t Index, float *pPlane) override;
    MTL_HRESULT SetRenderState(uint32_t State, uint32_t Value) override;
    MTL_HRESULT GetRenderState(uint32_t State, uint32_t *pValue) override;
    MTL_HRESULT CreateStateBlock(uint32_t Type, IMtlDirect3DStateBlock9 **ppSB) override;
    MTL_HRESULT BeginStateBlock() override;
    MTL_HRESULT EndStateBlock(IMtlDirect3DStateBlock9 **ppSB) override;
    MTL_HRESULT SetClipStatus(const void *) override { return D3D_S_OK; }
    MTL_HRESULT GetClipStatus(void *) override { return D3D_S_OK; }
    MTL_HRESULT GetTexture(uint32_t Stage, IMtlDirect3DBaseTexture9 **ppTexture) override;
    MTL_HRESULT SetTexture(uint32_t Stage, IMtlDirect3DBaseTexture9 *pTexture) override;
    MTL_HRESULT GetTextureStageState(uint32_t Stage, uint32_t Type, uint32_t *pValue) override;
    MTL_HRESULT SetTextureStageState(uint32_t Stage, uint32_t Type, uint32_t Value) override;
    MTL_HRESULT GetSamplerState(uint32_t Sampler, uint32_t Type, uint32_t *pValue) override;
    MTL_HRESULT SetSamplerState(uint32_t Sampler, uint32_t Type, uint32_t Value) override;
    MTL_HRESULT ValidateDevice(uint32_t *pNumPasses) override { if (pNumPasses) *pNumPasses = 1; return D3D_S_OK; }
    MTL_HRESULT SetPaletteEntries(uint32_t, const void *) override { return D3D_S_OK; }
    MTL_HRESULT GetPaletteEntries(uint32_t, void *) override { return D3D_S_OK; }
    MTL_HRESULT SetCurrentTexturePalette(uint32_t) override { return D3D_S_OK; }
    MTL_HRESULT GetCurrentTexturePalette(uint32_t *p) override { if (p) *p = 0; return D3D_S_OK; }
    MTL_HRESULT SetScissorRect(const D3DMtlRect *pRect) override;
    MTL_HRESULT GetScissorRect(D3DMtlRect *pRect) override;
    MTL_HRESULT SetSoftwareVertexProcessing(uint32_t) override { return D3D_S_OK; }
    uint32_t    GetSoftwareVertexProcessing() override { return 0; }
    MTL_HRESULT SetNPatchMode(float) override { return D3D_S_OK; }
    float       GetNPatchMode() override { return 0.0f; }
    MTL_HRESULT DrawPrimitive(uint32_t PrimitiveType, uint32_t StartVertex, uint32_t PrimitiveCount) override;
    MTL_HRESULT DrawIndexedPrimitive(uint32_t PrimitiveType, int32_t BaseVertexIndex, uint32_t MinVertexIndex, uint32_t NumVertices, uint32_t StartIndex, uint32_t PrimCount) override;
    MTL_HRESULT DrawPrimitiveUP(uint32_t PrimitiveType, uint32_t PrimitiveCount, const void *pVertexStreamZeroData, uint32_t VertexStreamZeroStride) override;
    MTL_HRESULT DrawIndexedPrimitiveUP(uint32_t PrimitiveType, uint32_t MinVertexIndex, uint32_t NumVertices, uint32_t PrimitiveCount, const void *pIndexData, uint32_t IndexDataFormat, const void *pVertexStreamZeroData, uint32_t VertexStreamZeroStride) override;
    MTL_HRESULT ProcessVertices(uint32_t, uint32_t, uint32_t, IMtlDirect3DVertexBuffer9 *, IMtlDirect3DVertexDeclaration9 *, uint32_t) override { return D3D_ERR_INVALIDCALL; }
    MTL_HRESULT CreateVertexDeclaration(const D3DMtlVertexElement *pVertexElements, IMtlDirect3DVertexDeclaration9 **ppDecl) override;
    MTL_HRESULT SetVertexDeclaration(IMtlDirect3DVertexDeclaration9 *pDecl) override;
    MTL_HRESULT GetVertexDeclaration(IMtlDirect3DVertexDeclaration9 **ppDecl) override;
    MTL_HRESULT SetFVF(uint32_t FVF) override;
    MTL_HRESULT GetFVF(uint32_t *pFVF) override;
    MTL_HRESULT CreateVertexShader(const uint32_t *pFunction, IMtlDirect3DVertexShader9 **ppShader) override;
    MTL_HRESULT SetVertexShader(IMtlDirect3DVertexShader9 *pShader) override;
    MTL_HRESULT GetVertexShader(IMtlDirect3DVertexShader9 **ppShader) override;
    MTL_HRESULT SetVertexShaderConstantF(uint32_t StartRegister, const float *pConstantData, uint32_t Vector4fCount) override;
    MTL_HRESULT GetVertexShaderConstantF(uint32_t StartRegister, float *pConstantData, uint32_t Vector4fCount) override;
    MTL_HRESULT SetVertexShaderConstantI(uint32_t StartRegister, const int32_t *pConstantData, uint32_t Vector4iCount) override;
    MTL_HRESULT GetVertexShaderConstantI(uint32_t StartRegister, int32_t *pConstantData, uint32_t Vector4iCount) override;
    MTL_HRESULT SetVertexShaderConstantB(uint32_t StartRegister, const uint32_t *pConstantData, uint32_t BoolCount) override;
    MTL_HRESULT GetVertexShaderConstantB(uint32_t StartRegister, uint32_t *pConstantData, uint32_t BoolCount) override;
    MTL_HRESULT SetStreamSource(uint32_t StreamNumber, IMtlDirect3DVertexBuffer9 *pStreamData, uint32_t OffsetInBytes, uint32_t Stride) override;
    MTL_HRESULT GetStreamSource(uint32_t StreamNumber, IMtlDirect3DVertexBuffer9 **ppStreamData, uint32_t *pOffsetInBytes, uint32_t *pStride) override;
    MTL_HRESULT SetStreamSourceFreq(uint32_t StreamNumber, uint32_t Setting) override;
    MTL_HRESULT GetStreamSourceFreq(uint32_t StreamNumber, uint32_t *pSetting) override;
    MTL_HRESULT SetIndices(IMtlDirect3DIndexBuffer9 *pIndexData) override;
    MTL_HRESULT GetIndices(IMtlDirect3DIndexBuffer9 **ppIndexData) override;
    MTL_HRESULT CreatePixelShader(const uint32_t *pFunction, IMtlDirect3DPixelShader9 **ppShader) override;
    MTL_HRESULT SetPixelShader(IMtlDirect3DPixelShader9 *pShader) override;
    MTL_HRESULT GetPixelShader(IMtlDirect3DPixelShader9 **ppShader) override;
    MTL_HRESULT SetPixelShaderConstantF(uint32_t StartRegister, const float *pConstantData, uint32_t Vector4fCount) override;
    MTL_HRESULT GetPixelShaderConstantF(uint32_t StartRegister, float *pConstantData, uint32_t Vector4fCount) override;
    MTL_HRESULT SetPixelShaderConstantI(uint32_t StartRegister, const int32_t *pConstantData, uint32_t Vector4iCount) override;
    MTL_HRESULT GetPixelShaderConstantI(uint32_t StartRegister, int32_t *pConstantData, uint32_t Vector4iCount) override;
    MTL_HRESULT SetPixelShaderConstantB(uint32_t StartRegister, const uint32_t *pConstantData, uint32_t BoolCount) override;
    MTL_HRESULT GetPixelShaderConstantB(uint32_t StartRegister, uint32_t *pConstantData, uint32_t BoolCount) override;
    MTL_HRESULT DrawRectPatch(uint32_t, const float *, const void *) override { return D3D_ERR_INVALIDCALL; }
    MTL_HRESULT DrawTriPatch(uint32_t, const float *, const void *) override { return D3D_ERR_INVALIDCALL; }
    MTL_HRESULT DeletePatch(uint32_t) override { return D3D_ERR_INVALIDCALL; }
    MTL_HRESULT CreateQuery(uint32_t Type, IMtlDirect3DQuery9 **ppQuery) override;

    /* Internal helpers for resources, swapchain, and stateblocks */
    id<MTLDevice>      mtlDevice() const { return mtlDevice_; }
    MtlRingBuffer     &ringBuffer() { return *ringBuffer_; }
    MtlPipelineCache  &pipelineCache() { return *pipelineCache_; }
    MtlDeviceState    &state() { return state_; }
    bool               isAppleGpu() const { return isAppleGpu_; }

    void enqueueTextureUpload(const PendingTextureUpload &up);
    void enqueueGenerateMipmaps(id<MTLTexture> tex);
    void readbackTextureToCpu(id<MTLTexture> tex, uint32_t level, uint32_t slice,
                              uint32_t d3dFormat, uint32_t w, uint32_t h,
                              std::vector<uint8_t> &out);
    void registerPendingClearSurface(MtlSurface9 *s);
    void unregisterPendingClearSurface(MtlSurface9 *s);
    void flushSurfacePendingClear(MtlSurface9 *s);

    /* Shared MTLBuffer recycling pool for dynamic/temporary VBs & IBs */
    const std::shared_ptr<MtlSharedBufferPool> &sharedBufferPool() const { return sharedBufferPool_; }

    /* Queries */
    uint32_t beginOcclusionQuery();
    uint64_t endOcclusionQuery(uint32_t slot);
    uint64_t issueEventQuery();
    uint64_t currentCommandSeq() const { return currentCmdSeq_; }
    uint64_t lastCommittedSeq() const { return lastCommittedSeq_.load(std::memory_order_acquire); }
    uint64_t completedCommandSeq() const { return completedCmdSeq_.load(std::memory_order_acquire); }
    uint64_t readVisibilityResult(uint32_t slot) const;
    void     flushCommands(bool waitUntilCompleted);
    bool     waitForCommandSeq(uint64_t targetSeq, uint32_t timeoutMs);
    bool     perfLogEnabled() const { return perfLogEnabled_; }
    void     recordPerfQueryWait(uint64_t waitNs, bool isOcclusion, bool hadFlushFlag, bool triggeredFlush, bool missed) {
        if (!perfLogEnabled_) return;
        perfQueryWaitNsAcc_.fetch_add(waitNs, std::memory_order_relaxed);
        if (isOcclusion) perfOccQueryWaitNsAcc_.fetch_add(waitNs, std::memory_order_relaxed);
        else perfEvtQueryWaitNsAcc_.fetch_add(waitNs, std::memory_order_relaxed);
        if (hadFlushFlag) perfQueryFlushFlagAcc_.fetch_add(1u, std::memory_order_relaxed);
        if (triggeredFlush) perfQueryTrigFlushAcc_.fetch_add(1u, std::memory_order_relaxed);
        if (missed) perfQueryMissAcc_.fetch_add(1u, std::memory_order_relaxed);
    }

    /* SwapChain Present entry */
    MTL_HRESULT presentSwapChain(MtlSwapChain9 *sc, const D3DMtlRect *srcRect,
                                 const D3DMtlRect *dstRect, uint32_t flags);
    void onWarmSetMap(const char *name);

private:
    /* Command buffer & TBDR render pass management */
    void ensureCommandBuffer();
    void maybeSplitAfterOcclusionBatch(uint32_t minEnded = 16u);
    void flushPendingBlits();
    void flushPendingClearsForSampledTextures(uint32_t vsSampMask, uint32_t psSampMask);
    void endRenderPass(bool discardDepth = false, bool discardStencil = false, uint8_t discardColorMask = 0);
    bool beginRenderPassIfNeeded(bool allowColorForDraw = true);
    void commitCommandBuffer(bool waitUntilCompleted, bool isPresentFrame, id<CAMetalDrawable> drawableToPresent, bool useVsync);

    /* Draw preparation */
    uint8_t activeColorWriteRtMask() const;
    bool    canUseFullScreenDontCareBase(uint32_t primType, uint32_t primCount, uint32_t minStride,
                                         int &outPosOff, bool &outIsClipSpace, uint32_t &outPosDeclType,
                                         uint32_t &outRtW, uint32_t &outRtH) const;
    void    tryMarkFullScreenDontCare(uint32_t primType, uint32_t primCount,
                                      const void *vbData, uint32_t stride, uint32_t vertCount);
    void    tryMarkFullScreenDontCareIndexed(uint32_t primType, int32_t baseVertexIndex,
                                             uint32_t startIndex, uint32_t primCount,
                                             MtlVertexBuffer9 *vb, uint32_t vbByteOffset,
                                             uint32_t stride, MtlIndexBuffer9 *ib);
    bool prepareDraw(uint32_t primitiveType, bool useUpStream0, id<MTLBuffer> upVb, size_t upVbOffset,
                     uint32_t upStride, const MtlCompiledShader **outPs);
    MtlVertexDeclaration9 *getDeclForFvf(uint32_t fvf);
    void blitQuad(id<MTLTexture> srcTex, const D3DMtlRect *srcRect, uint32_t srcW, uint32_t srcH,
                  id<MTLTexture> dstTex, uint32_t dstLevel, uint32_t dstSlice,
                  MTLPixelFormat dstFmt, const D3DMtlRect *dstRect, uint32_t dstW, uint32_t dstH,
                  bool linearFilter);
    void clearRectQuad(MtlSurface9 *rt, MtlSurface9 *ds, const D3DMtlRect &rect,
                       uint32_t flags, D3DCOLOR_T color, float z, uint32_t stencil);

    MtlFastRecursiveLock mu_;

    IMtlDirect3D9 *d3d9_;
    id<MTLDevice> mtlDevice_;
    id<MTLCommandQueue> cmdQueue_ = nil;
    bool isAppleGpu_ = true;

    D3DMtlCreationParameters creationParams_ = {};
    D3DMtlPresentParameters  presentParams_ = {};
    D3DMtlGammaRamp          gammaRamp_ = {};

    std::unique_ptr<MtlRingBuffer>     ringBuffer_;
    std::unique_ptr<MtlShaderCompiler> shaderCompiler_;
    std::unique_ptr<MtlPipelineCache>  pipelineCache_;

    MtlSwapChain9 *primarySwapChain_ = nullptr;
    MtlSurface9   *autoDepthStencil_ = nullptr;
    MtlSurface9   *renderTargets_[4] = {};
    MtlSurface9   *depthStencil_ = nullptr;

    MtlDeviceState  state_;
    MtlStateBlock9 *recordingBlock_ = nullptr;

    std::unordered_map<uint32_t, MtlVertexDeclaration9 *> fvfDeclCache_;

    /* Frame pacing (triple buffered in-flight frames) */
    dispatch_semaphore_t frameSem_ = nullptr;
    uint32_t frameIndex_ = 0;

    /* Active Metal command buffer & render pass state */
    id<MTLCommandBuffer>        cmdBuf_ = nil;
    id<MTLRenderCommandEncoder> encoder_ = nil;
    uint64_t                    currentCmdSeq_ = 1;
    std::atomic<uint64_t>       lastCommittedSeq_{0};
    std::atomic<uint64_t>       completedCmdSeq_{0};
    std::mutex                  seqWaitMu_;
    std::condition_variable     seqWaitCv_;
    bool                        hasPendingWork_ = false;

    /* Pending clear coalescing for TBDR MTLLoadActionClear */
    std::vector<MtlSurface9 *> pendingClearSurfaces_;
    bool       pendingColorDontCare_ = false;

    /* Active pass tracking & redundant-bind filtering */
    bool     passSrgbWrite_ = false;
    uint32_t passWidth_ = 0;
    uint32_t passHeight_ = 0;
    bool     passHasColorAttach_ = false;
    uint8_t  passColorMask_ = 0;
    bool     passHasDepthAttach_ = false;
    bool     passHasStencilAttach_ = false;
    bool     passHasVisBuffer_ = false;
    MtlSurface9 *passColorSurfaces_[4] = {};
    MtlSurface9 *passDsSurface_ = nullptr;
    id<MTLRenderPipelineState> boundPso_ = nil;
    id<MTLDepthStencilState>   boundDss_ = nil;
    const MtlCompiledShader   *boundPsCompiled_ = nullptr;
    uint8_t  boundActiveRtMask_ = 0;
    bool     boundAttachDs_ = false;
    bool     boundPsPointSpriteEnable_ = false;
    uint16_t boundPsDepthCmpMask_ = 0;
    uint16_t boundPsNullMask_ = 0;
    uint16_t boundPsProjMask_ = 0;
    uint16_t boundPsGatherMask_ = 0;
    bool     psoDirty_ = true;
    bool     dssDirty_ = true;
    bool     vpDirty_ = true;
    bool     scissorDirty_ = true;
    bool     cullDirty_ = true;
    bool     fillDirty_ = true;
    bool     slopeDirty_ = true;
    bool     blendColorDirty_ = true;
    bool     stencilRefDirty_ = true;
    bool     frontFaceBound_ = false;
    bool     zeroAttrBound_ = false;
    uint16_t streamDirtyMask_ = 0xFFFFu;
    uint16_t psTexDirtyMask_ = 0xFFFFu;
    uint16_t psSampDirtyMask_ = 0xFFFFu;
    id<MTLBuffer>       boundVb_[16] = {};
    size_t              boundVbOff_[16] = {};
    id<MTLTexture>      boundFragTex_[16] = {};
    id<MTLSamplerState> boundFragSmp_[16] = {};
    uint32_t vsConstFVersion_ = 1;
    uint32_t psConstFVersion_ = 1;
    uint32_t boundVsConstFVersion_ = 0;
    uint32_t boundPsConstFVersion_ = 0;
    uint16_t boundVsConstFVecs_ = 0;
    uint16_t boundPsConstFVecs_ = 0;
    bool     vsConstIDirty_ = true;
    bool     vsConstBDirty_ = true;
    bool     psConstIDirty_ = true;
    bool     psConstBDirty_ = true;
    bool     vsFogDirty_ = true;
    bool     psFogDirty_ = true;
    bool     psTexStageDirty_ = true;
    uint32_t boundPsSampCbvMask_ = 0;
    bool     perfLogEnabled_ = false;
    uint64_t perfFrames_ = 0;
    double   perfCpuMsAcc_ = 0.0;
    double   perfWaitMsAcc_ = 0.0;
    double   perfGpuMsAcc_ = 0.0;
    uint64_t perfDrawsAcc_ = 0;
    std::atomic<uint64_t> perfGpuUsTotal_{0};
    std::atomic<uint32_t> perfGpuSamples_{0};
    std::atomic<uint64_t> perfQueryWaitNsAcc_{0};
    std::atomic<uint64_t> perfOccQueryWaitNsAcc_{0};
    std::atomic<uint64_t> perfEvtQueryWaitNsAcc_{0};
    std::atomic<uint32_t> perfQueryFlushFlagAcc_{0};
    std::atomic<uint32_t> perfQueryTrigFlushAcc_{0};
    std::atomic<uint32_t> perfQueryMissAcc_{0};
    uint64_t perfCommitsAcc_ = 0;
    uint64_t perfVisSplitsAcc_ = 0;

    bool     gpuPassLogEnabled_ = false;
    bool     gpuPassProfileActive_ = false;
    uint64_t gpuPassFrameCounter_ = 0;
    uint32_t gpuPassIndex_ = 0;
    double   gpuPassTotalMs_ = 0.0;
    struct GpuPassSnapshot {
        uint32_t w = 0, h = 0;
        uint32_t rtFmt[4] = {};
        uint8_t  rtClear[4] = {};
        uint32_t dsFmt = 0;
        uint8_t  zClear = 0;
        uint8_t  sClear = 0;
        uint8_t  discardD = 0;
        uint8_t  discardS = 0;
        uint32_t draws = 0;
        std::unordered_map<uint64_t, uint32_t> psCounts;
    } curGpuPass_;

    void invalidatePassBindCache();

    /* Pending uploads & mipgen queued before next render pass */
    std::vector<PendingTextureUpload> pendingUploads_;
    std::vector<id<MTLTexture>>       pendingMipGens_;

    /* Recycled shared MTLBuffers for dynamic VBs & IBs */
    std::shared_ptr<MtlSharedBufferPool> sharedBufferPool_;

    /* Visibility / occlusion query ring */
    id<MTLBuffer> visBuffer_ = nil;
    uint64_t     *visPtr_ = nullptr;
    uint32_t      nextVisSlot_ = 0;
    uint32_t      activeVisSlot_ = kInvalidVisSlot;
    uint32_t      occEndedInCmdBuf_ = 0;
    uint32_t      occSplitsThisFrame_ = 0;
    bool          hasAnyOcclusionQuery_ = false;

    /* Per-map PSO warmup & stall telemetry */
    std::atomic<uint64_t> currentMapHash_{0};
    std::atomic<uint64_t> activeWarmGen_{0};
    std::atomic<bool>     warmStop_{false};
    std::atomic<bool>     allMapsWarmed_{false};
    uint64_t recordedManifestMapHash_ = 0;
    std::unordered_set<MtlRenderPipelineKey, MtlRenderPipelineKeyHash> recordedManifestPsoKeys_;

    bool stallLogEnabled_ = false;
    bool hasLastPresentTime_ = false;
    std::chrono::steady_clock::time_point lastPresentTime_{};
    std::atomic<uint32_t> stallVsCreateCnt_{0};
    std::atomic<uint64_t> stallVsCreateNs_{0};
    std::atomic<uint32_t> stallPsCreateCnt_{0};
    std::atomic<uint64_t> stallPsCreateNs_{0};
    uint32_t stallPsVarMissCnt_ = 0;
    double   stallPsVarMissMs_ = 0.0;
    uint32_t stallPsoHitCnt_ = 0;
    double   stallPsoHitMs_ = 0.0;
    uint32_t stallPsoMissCnt_ = 0;
    double   stallPsoMissMs_ = 0.0;
    uint32_t stallBgPsHitCnt_ = 0;
    uint32_t stallWarmPsoHitCnt_ = 0;
};

/* Helper: map D3D9 sampler index (0..15 or 257..260) -> internal slot 0..19, or -1 */
inline int mapD3dSamplerSlot(uint32_t d3dSampler) {
    if (d3dSampler < 16u) return int(d3dSampler);
    if (d3dSampler >= D3DM_VERTEXTEXTURESAMPLER0 && d3dSampler < D3DM_VERTEXTEXTURESAMPLER0 + 4u) {
        return int(16u + (d3dSampler - D3DM_VERTEXTEXTURESAMPLER0));
    }
    return -1;
}

} // namespace mtl_d3d9
