import { Builder, ByteBuffer, SIZE_PREFIX_LENGTH } from "flatbuffers";
import * as Generated from "./ndscenepy/flatbuffers/generated/ndscene.js";
import { NDPacketRoot } from "./ndscenepy/flatbuffers/generated/ndscene/ndpacket-root.js";
export * from "./ndscenepy/flatbuffers/generated/ndscene.js";
export { Builder, ByteBuffer, SIZE_PREFIX_LENGTH };
export function ndPacketBytes(input) {
    if (input instanceof Uint8Array) {
        return input;
    }
    if (ArrayBuffer.isView(input)) {
        return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    return new Uint8Array(input);
}
export function ndPacketByteBuffer(input) {
    return new ByteBuffer(ndPacketBytes(input));
}
export function hasNDPacketRootIdentifier(input, options = {}) {
    const byteBuffer = ndPacketByteBuffer(input);
    if (options.sizePrefixed) {
        byteBuffer.setPosition(SIZE_PREFIX_LENGTH);
    }
    return NDPacketRoot.bufferHasIdentifier(byteBuffer);
}
export function ndPacketRootFromBuffer(input, options = {}) {
    const { sizePrefixed = false, validateIdentifier = true, root } = options;
    if (validateIdentifier && !hasNDPacketRootIdentifier(input, { sizePrefixed })) {
        throw new Error("Buffer does not contain an NDPacketRoot with the NDSN file identifier.");
    }
    const byteBuffer = ndPacketByteBuffer(input);
    return sizePrefixed
        ? NDPacketRoot.getSizePrefixedRootAsNDPacketRoot(byteBuffer, root)
        : NDPacketRoot.getRootAsNDPacketRoot(byteBuffer, root);
}
export const ndsceneFlatbuffers = Generated;
