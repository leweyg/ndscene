"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ndsceneFlatbuffers = exports.SIZE_PREFIX_LENGTH = exports.ByteBuffer = void 0;
exports.ndPacketBytes = ndPacketBytes;
exports.ndPacketByteBuffer = ndPacketByteBuffer;
exports.hasNDPacketRootIdentifier = hasNDPacketRootIdentifier;
exports.ndPacketRootFromBuffer = ndPacketRootFromBuffer;
const flatbuffers_1 = require("flatbuffers");
Object.defineProperty(exports, "ByteBuffer", { enumerable: true, get: function () { return flatbuffers_1.ByteBuffer; } });
Object.defineProperty(exports, "SIZE_PREFIX_LENGTH", { enumerable: true, get: function () { return flatbuffers_1.SIZE_PREFIX_LENGTH; } });
const Generated = __importStar(require("./ndscenepy/flatbuffers/generated/ndscene.js"));
const ndpacket_root_js_1 = require("./ndscenepy/flatbuffers/generated/ndscene/ndpacket-root.js");
__exportStar(require("./ndscenepy/flatbuffers/generated/ndscene.js"), exports);
function ndPacketBytes(input) {
    if (input instanceof Uint8Array) {
        return input;
    }
    if (ArrayBuffer.isView(input)) {
        return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    return new Uint8Array(input);
}
function ndPacketByteBuffer(input) {
    return new flatbuffers_1.ByteBuffer(ndPacketBytes(input));
}
function hasNDPacketRootIdentifier(input, options = {}) {
    const byteBuffer = ndPacketByteBuffer(input);
    if (options.sizePrefixed) {
        byteBuffer.setPosition(flatbuffers_1.SIZE_PREFIX_LENGTH);
    }
    return ndpacket_root_js_1.NDPacketRoot.bufferHasIdentifier(byteBuffer);
}
function ndPacketRootFromBuffer(input, options = {}) {
    const { sizePrefixed = false, validateIdentifier = true, root } = options;
    if (validateIdentifier && !hasNDPacketRootIdentifier(input, { sizePrefixed })) {
        throw new Error("Buffer does not contain an NDPacketRoot with the NDSN file identifier.");
    }
    const byteBuffer = ndPacketByteBuffer(input);
    return sizePrefixed
        ? ndpacket_root_js_1.NDPacketRoot.getSizePrefixedRootAsNDPacketRoot(byteBuffer, root)
        : ndpacket_root_js_1.NDPacketRoot.getRootAsNDPacketRoot(byteBuffer, root);
}
exports.ndsceneFlatbuffers = Generated;
