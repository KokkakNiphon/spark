// @ts-nocheck
import * as THREE from "three";
import {
  Fn,
  abs,
  acos,
  add,
  all,
  any,
  atan,
  attribute,
  bool,
  clamp,
  cos,
  cross,
  Discard,
  div,
  dot,
  equal,
  exp,
  float,
  floor,
  fract,
  If,
  instanceIndex,
  int,
  log,
  log2,
  max,
  min,
  mix,
  mod,
  mul,
  negate,
  normalize,
  not,
  or,
  positionLocal,
  positionWorld,
  pow,
  round,
  sin,
  sqrt,
  sub,
  tan,
  texture,
  transpose,
  uint,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
  mat2,
  mat3,
  mat4,
  uvec2,
  uvec3,
  uvec4,
  ivec3,
  positionGeometry,
  cameraProjectionMatrix,
  cameraViewMatrix,
  Loop,
  shiftLeft as shl,
  shiftRight as shr,
  bitAnd,
  bitOr,
  bitXor,
  greaterThan,
  greaterThanEqual,
  lessThan,
  lessThanEqual,
  and,
  bitcast,
  screenCoordinate,
  varying,
  glsl
} from "three/tsl";

// Constants
export const LN_SCALE_MIN = float(-12.0);
export const LN_SCALE_MAX = float(9.0);

const SPLAT_TEX_WIDTH_BITS = uint(11);
const SPLAT_TEX_HEIGHT_BITS = uint(11);
const SPLAT_TEX_LAYER_BITS = add(SPLAT_TEX_WIDTH_BITS, SPLAT_TEX_HEIGHT_BITS);

const SPLAT_TEX_WIDTH = shl(uint(1), SPLAT_TEX_WIDTH_BITS);
const SPLAT_TEX_HEIGHT = shl(uint(1), SPLAT_TEX_HEIGHT_BITS);

const SPLAT_TEX_WIDTH_MASK = sub(SPLAT_TEX_WIDTH, uint(1));
const SPLAT_TEX_HEIGHT_MASK = sub(SPLAT_TEX_HEIGHT, uint(1));

// Utility Functions

const sqr = Fn(([x]) => mul(x, x));

const decodeQuatOctXy88R8 = Fn(([encoded]) => {
    const quantU = bitAnd(encoded, uint(0xFF));
    const quantV = bitAnd(shr(encoded, uint(8)), uint(0xFF));
    const angleInt = shr(encoded, uint(16));

    const u_f = div(float(quantU), 255.0);
    const v_f = div(float(quantV), 255.0);

    const f = vec2(sub(mul(u_f, 2.0), 1.0), sub(mul(v_f, 2.0), 1.0));

    const axis = vec3(f.x, f.y, sub(sub(1.0, abs(f.x)), abs(f.y))).toVar();

    const t = max(negate(axis.z), 0.0);

    If (greaterThanEqual(axis.x, 0.0), () => {
        axis.x.addAssign(negate(t));
    }).Else(() => {
        axis.x.addAssign(t);
    });

    If (greaterThanEqual(axis.y, 0.0), () => {
        axis.y.addAssign(negate(t));
    }).Else(() => {
        axis.y.addAssign(t);
    });

    const normalizedAxis = normalize(axis);

    const theta = mul(div(float(angleInt), 255.0), 3.14159265359);
    const halfTheta = mul(theta, 0.5);
    const s = sin(halfTheta);
    const w = cos(halfTheta);

    return vec4(mul(normalizedAxis, s), w);
});

const unpackSplatEncoding = Fn(([packed, rgbMinMaxLnScaleMinMax]) => {
    const word0 = packed.x;
    const word1 = packed.y;
    const word2 = packed.z;
    const word3 = packed.w;

    const uRgba = uvec4(
        bitAnd(word0, uint(0xff)),
        bitAnd(shr(word0, uint(8)), uint(0xff)),
        bitAnd(shr(word0, uint(16)), uint(0xff)),
        bitAnd(shr(word0, uint(24)), uint(0xff))
    );

    const rgbMin = rgbMinMaxLnScaleMinMax.x;
    const rgbMax = rgbMinMaxLnScaleMinMax.y;

    const rgbaRaw = div(vec4(uRgba), 255.0);
    const rgb = add(mul(rgbaRaw.rgb, sub(rgbMax, rgbMin)), rgbMin);
    const rgba = vec4(rgb, rgbaRaw.a);

    // unpackHalf2x16 equivalent
    const centerXY = uint(word1).unpack2x16float();
    const centerZ_ = uint(bitAnd(word2, uint(0xffff))).unpack2x16float();
    const center = vec3(centerXY, centerZ_.x);

    const uScales = uvec3(
        bitAnd(word3, uint(0xff)),
        bitAnd(shr(word3, uint(8)), uint(0xff)),
        bitAnd(shr(word3, uint(16)), uint(0xff))
    );

    const lnScaleMin = rgbMinMaxLnScaleMinMax.z;
    const lnScaleMax = rgbMinMaxLnScaleMinMax.w;
    const lnScaleScale = div(sub(lnScaleMax, lnScaleMin), 254.0);

    const scales = vec3(
        equal(uScales.x, uint(0)).select(float(0.0), exp(add(lnScaleMin, mul(float(sub(uScales.x, uint(1))), lnScaleScale)))),
        equal(uScales.y, uint(0)).select(float(0.0), exp(add(lnScaleMin, mul(float(sub(uScales.y, uint(1))), lnScaleScale)))),
        equal(uScales.z, uint(0)).select(float(0.0), exp(add(lnScaleMin, mul(float(sub(uScales.z, uint(1))), lnScaleScale))))
    );

    const uQuat = bitOr(
        bitAnd(shr(word2, uint(16)), uint(0xFFFF)),
        bitAnd(shr(word3, uint(8)), uint(0xFF0000))
    );
    const quaternion = decodeQuatOctXy88R8(uQuat);

    return { center, scales, quaternion, rgba };
});

const quatVec = Fn(([q, v]) => {
    const t = mul(2.0, cross(q.xyz, v));
    return add(add(v, mul(q.w, t)), cross(q.xyz, t));
});

const quatQuatCorrect = Fn(([q1, q2]) => {
    return vec4(
        add(sub(add(mul(q1.w, q2.x), mul(q1.x, q2.w)), mul(q1.y, q2.z)), mul(q1.z, q2.y)),
        add(add(sub(mul(q1.w, q2.y), mul(q1.x, q2.z)), mul(q1.y, q2.w)), mul(q1.z, q2.x)),
        add(sub(add(mul(q1.w, q2.z), mul(q1.x, q2.y)), mul(q1.y, q2.x)), mul(q1.z, q2.w)),
        sub(sub(sub(mul(q1.w, q2.w), mul(q1.x, q2.x)), mul(q1.y, q2.y)), mul(q1.z, q2.z))
    );
});


const scaleQuaternionToMatrix = Fn(([s, q]) => {
    const xx = mul(q.x, q.x);
    const yy = mul(q.y, q.y);
    const zz = mul(q.z, q.z);
    const xy = mul(q.x, q.y);
    const xz = mul(q.x, q.z);
    const yz = mul(q.y, q.z);
    const wx = mul(q.w, q.x);
    const wy = mul(q.w, q.y);
    const wz = mul(q.w, q.z);

    // Column-major construction for mat3
    // Use 9 arguments to be safe
    return mat3(
        mul(s.x, sub(1.0, mul(2.0, add(yy, zz)))),
        mul(s.y, mul(2.0, sub(xy, wz))),
        mul(s.z, mul(2.0, add(xz, wy))),

        mul(s.x, mul(2.0, add(xy, wz))),
        mul(s.y, sub(1.0, mul(2.0, add(xx, zz)))),
        mul(s.z, mul(2.0, sub(yz, wx))),

        mul(s.x, mul(2.0, sub(xz, wy))),
        mul(s.y, mul(2.0, add(yz, wx))),
        mul(s.z, sub(1.0, mul(2.0, add(xx, yy))))
    );
});

export const splatVertex = Fn(([
    renderSize, numSplats, renderToViewQuat, renderToViewPos,
    maxStdDev, minPixelRadius, maxPixelRadius, minAlpha,
    stochastic, enable2DGS, blurAmount, preBlurAmount,
    focalDistance, apertureAngle, clipXY, focalAdjustment,
    packedSplats, rgbMinMaxLnScaleMinMax
]) => {
    const splatIndex = attribute('splatIndex', 'uint');
    const instanceId = instanceIndex;

    Discard(greaterThanEqual(instanceId, numSplats));

    const texCoord = uvec3(0, 0, 0).toVar();

    If (stochastic, () => {
        const x = bitAnd(instanceId, SPLAT_TEX_WIDTH_MASK);
        const y = bitAnd(shr(instanceId, SPLAT_TEX_WIDTH_BITS), SPLAT_TEX_HEIGHT_MASK);
        const z = shr(instanceId, SPLAT_TEX_LAYER_BITS);
        texCoord.assign(uvec3(x, y, z));
    }).Else(() => {
        If(equal(splatIndex, uint(0xffffffff)), () => {
            Discard();
        });
        const x = bitAnd(splatIndex, SPLAT_TEX_WIDTH_MASK);
        const y = bitAnd(shr(splatIndex, SPLAT_TEX_WIDTH_BITS), SPLAT_TEX_HEIGHT_MASK);
        const z = shr(splatIndex, SPLAT_TEX_LAYER_BITS);
        texCoord.assign(uvec3(x, y, z));
    });

    const packed = texture(packedSplats).load(texCoord);

    const unpacked = unpackSplatEncoding(packed, rgbMinMaxLnScaleMinMax);
    const center = unpacked.center;
    const scales = unpacked.scales;
    const quaternion = unpacked.quaternion;
    const rgba = unpacked.rgba;

    Discard(lessThan(rgba.a, minAlpha));

    const zeroScales = equal(scales, vec3(0.0));
    Discard(all(zeroScales));

    const viewCenter = add(quatVec(renderToViewQuat, center), renderToViewPos);

    Discard(greaterThanEqual(viewCenter.z, 0.0));

    const clipCenter = mul(cameraProjectionMatrix, vec4(viewCenter, 1.0));

    Discard(greaterThanEqual(abs(clipCenter.z), clipCenter.w));

    const clipVal = mul(clipXY, clipCenter.w);
    Discard(or(greaterThan(abs(clipCenter.x), clipVal), greaterThan(abs(clipCenter.y), clipVal)));

    const vSplatIndex = varying(stochastic.select(instanceId, splatIndex));

    const viewQuaternion = quatQuatCorrect(renderToViewQuat, quaternion);

    const vRgba = varying(rgba);
    const vSplatUv = varying(vec2(0.0));
    const vNdc = varying(vec3(0.0));
    const glPosition = vec4(0.0).toVar();

    const posLocal = positionGeometry.xy;

    If (and(enable2DGS, any(zeroScales)), () => {
        vRgba.assign(rgba);
        vSplatUv.assign(mul(posLocal, maxStdDev));

        const offset = vec3(0.0).toVar();
        If (zeroScales.z, () => {
            offset.assign(vec3(mul(vSplatUv.x, scales.x), mul(vSplatUv.y, scales.y), 0.0));
        }).ElseIf (zeroScales.y, () => {
            offset.assign(vec3(mul(vSplatUv.x, scales.x), 0.0, mul(vSplatUv.y, scales.z)));
        }).Else (() => {
            offset.assign(vec3(0.0, mul(vSplatUv.x, scales.y), mul(vSplatUv.y, scales.z)));
        });

        const viewPos = add(viewCenter, quatVec(viewQuaternion, offset));
        glPosition.assign(mul(cameraProjectionMatrix, vec4(viewPos, 1.0)));
        vNdc.assign(div(glPosition.xyz, glPosition.w));
    }).Else(() => {
        const ndcCenter = div(clipCenter.xyz, clipCenter.w);

        const RS = scaleQuaternionToMatrix(scales, viewQuaternion);
        const cov3D = mul(RS, transpose(RS));

        const scaledRenderSize = mul(renderSize, focalAdjustment);

        // Use int(0) and int(1) for element access
        const P00 = cameraProjectionMatrix.element(int(0)).x;
        const P11 = cameraProjectionMatrix.element(int(1)).y;

        const focal = mul(mul(0.5, scaledRenderSize), vec2(P00, P11));

        const invZ = div(1.0, viewCenter.z);
        const J1 = mul(focal, invZ);
        const J2 = mul(negate(mul(J1, viewCenter.xy)), invZ);

        const J_col0 = vec3(J1.x, 0.0, 0.0);
        const J_col1 = vec3(0.0, J1.y, 0.0);
        const J_col2 = vec3(J2.x, J2.y, 0.0);
        const J = mat3(J_col0, J_col1, J_col2);

        const cov2D = mul(mul(transpose(J), cov3D), J);

        const a = cov2D.element(int(0)).x.toVar();
        const d = cov2D.element(int(1)).y.toVar();
        const b = cov2D.element(int(0)).y;

        a.addAssign(preBlurAmount);
        d.addAssign(preBlurAmount);

        const fullBlurAmount = blurAmount.toVar();

        If (and(greaterThan(focalDistance, 0.0), greaterThan(apertureAngle, 0.0)), () => {
             const focusRadius = maxPixelRadius.toVar();
             If (lessThan(viewCenter.z, 0.0), () => {
                 const focusBlur = abs(div(sub(negate(viewCenter.z), focalDistance), viewCenter.z));
                 const apertureRadius = mul(focal.x, tan(mul(0.5, apertureAngle)));
                 focusRadius.assign(mul(focusBlur, apertureRadius));
             });
             fullBlurAmount.assign(clamp(sqr(focusRadius), blurAmount, sqr(maxPixelRadius)));
        });

        const detOrig = sub(mul(a, d), mul(b, b));
        a.addAssign(fullBlurAmount);
        d.addAssign(fullBlurAmount);
        const det = sub(mul(a, d), mul(b, b));

        const blurAdjust = sqrt(max(0.0, div(detOrig, det)));
        vRgba.a.mulAssign(blurAdjust);

        Discard(lessThan(vRgba.a, minAlpha));

        const eigenAvg = mul(0.5, add(a, d));
        const eigenDelta = sqrt(max(0.0, sub(mul(eigenAvg, eigenAvg), det)));
        const eigen1 = add(eigenAvg, eigenDelta);
        const eigen2 = sub(eigenAvg, eigenDelta);

        const eigenVec1 = normalize(vec2(lessThan(abs(b), 0.001).select(1.0, b), sub(eigen1, a)));
        const eigenVec2 = vec2(eigenVec1.y, negate(eigenVec1.x));

        const scale1 = min(maxPixelRadius, mul(maxStdDev, sqrt(eigen1)));
        const scale2 = min(maxPixelRadius, mul(maxStdDev, sqrt(eigen2)));

        Discard(and(lessThan(scale1, minPixelRadius), lessThan(scale2, minPixelRadius)));

        const pixelOffset = add(mul(mul(posLocal.x, eigenVec1), scale1), mul(mul(posLocal.y, eigenVec2), scale2));
        const ndcOffset = mul(div(2.0, scaledRenderSize), pixelOffset);
        const ndc = vec3(add(ndcCenter.xy, ndcOffset), ndcCenter.z);

        vSplatUv.assign(mul(posLocal, maxStdDev));
        vNdc.assign(ndc);
        glPosition.assign(vec4(mul(ndc.xy, clipCenter.w), clipCenter.zw));
    });

    return {
        glPosition,
        vRgba,
        vSplatUv,
        vNdc,
        vSplatIndex
    };
});

export const splatFragment = Fn(([
    maxStdDev, minAlpha, falloff, encodeLinear, stochastic, time,
    vRgba, vSplatUv, vNdc, vSplatIndex
]) => {
    const rgba = vRgba.toVar();
    const z = dot(vSplatUv, vSplatUv);

    Discard(greaterThan(z, sqr(maxStdDev)));

    rgba.a.mulAssign(mix(1.0, exp(mul(-0.5, z)), falloff));

    Discard(lessThan(rgba.a, minAlpha));

    If (encodeLinear, () => {
         rgba.rgb.assign(pow(rgba.rgb, vec3(2.2)));
    });

    If (stochastic, () => {
        const uTime = bitcast(time, uint);

        const coord = uvec2(uint(screenCoordinate.x), uint(screenCoordinate.y));

        const state = add(add(add(uTime, mul(uint(0x9e3779b9), coord.x)), mul(uint(0x85ebca6b), coord.y)), mul(uint(0xc2b2ae35), vSplatIndex)).toVar();
        state.assign(add(mul(state, uint(747796405)), uint(2891336453)));

        const hash = mul(bitXor(shr(state, add(shr(state, uint(28)), uint(4))), state), uint(277803737)).toVar();
        hash.assign(bitXor(shr(hash, uint(22)), hash));

        const rand = div(float(hash), 4294967296.0);

        Discard(greaterThanEqual(rand, rgba.a));
        rgba.a.assign(1.0);
    });

    return rgba;
});
