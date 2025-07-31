
# NDScene: NDObject, NDTensor and NDRender
n-dimensional scene-graph format and runtime

## Abstract / Architectural Principals

1. "Tensors" (NDTensor) should be recursivly nestable, i.e. tensors can be dictionaries of tensors. In this way most structured data is already considered to be in tensor format.
2. "Objects" (NDObject) are posed/transformed sparse tensors, equivalent to the concatenation of their tranformed children, including support for parallel dictionaries of tensors. This combines the flexibility of scene composition with the effeciency of dense/repeated tensors.
3. "Data" (NDData) behind tensors can be progressivly transitioned between tensored, buffered, mime-compressed and remote-pathed states. Allowing natural integration of standard image, video, zip and other compression schemes.
3. "Scenes" (NDScene) are collections of objects including support of reuse/sharing of Objects, Tensors and Data; which is useful in instancing and other techniques.
3. "Poses/Transforms" (PoseND) also being tensors default to being batch-matrix-multiply transforms along each parallel dictionary dimension, but can be replaced by reference to externally defined non-recurive-tensor "models".
4. "Rendering" (NDRender) is the process of walking a posed tree of concatented objects, and differentiably unposing their result back into an updated target objects state. Caching is handled either automatically or by careful walking/updating of the scene tree.
5. "Streaming" (StreamND) is achieved via scene patches/updates, including scenes which are themselves queries for additional content, and which generally leverage a secondary path-based file/shared-memory system for same-device or cacheable content.
6. "Hardware Acceleration" (NDTorch/etc.) should be optional and lazily imported to not cause delays/breaks where it's not used.

## JSON Schema

In brief: scenes contain objects which contain dictionaries of tensor-shaped data. Object transforms can be selection from shared tensors. The @type semantics below are optional and included for clarity.

```json
{
    // 
    "@type": "NDScene", // scene of nd-objects
    "root": "string_key|NDObject", // id of root object
    "paths" :{"string_key":{ // list or dictionary of objects

        "@type": "NDObject", // tensor-based scene element
        "key": "string_key", // name/id of this object
        "children": "list<string_key|NDObject", // move with this object
        "parents": "list<string_key|NDObject", // multiple for bundle adjustment
        "pose": "NDTensor", // read(): return pose * concat( data, children.read() )
        "unpose": "NDTensor", // write(value): data = unpose * value
        "data": {

            "@type" :"NDTensor", // nestable/recursivly-defined tensor
            "key" :"string", // the dimension name / dictionary key
            "size" :"int", // the size used in 'shape:[int]' shorthand
            "shape" :"list<NDTensor>", // nested for dictionaries
            "dtype" :"string_dtype", // data element type, traditionally an enum
            "data" : {
                
                "@type" : "NDData |array|bytes|string", // linear data representation
                "tensor" : ["number|string|numpy|pytorch"],
                "buffer" : "bytes",
                "buffer_size": "int", // size uncompressed
                "buffer_dtype": "string_dtype",
                "compressed" : "bytes",
                "compressed_type" : "string_mime_type",
                "compressed_size" : "int", 
                "path" : "string", // external path
            }
        }
    }}
}
```

For convenience these can be specified in pure JSON, and don't need to be strongly typed before being passed to the Render APIs. Unrecognized dictionary keys are considered shape properties (in the provided order), and shapes without named keys are considered sizes:

| JSON | NDTensor |
| --- | --- |
| `{shape:[2,3],data=[1,2,3,4,5,6]}` | `{shape:[{size:2},{size:3}],data:[1,2,3,4,5,6],dtype=number,size:6}` |
| `{foo:bar,etc:8}` | `{shape:[{key:foo,data:bar},{key:etc,data:8}]}` |


## API - Python

```python
class NDRender:
    stack_result :NDTensor = None
    stack_pose   :list[NDTensor] = None
    stack_values :list[NDTensor] = None

    # NDTensor/JSON conversions:
    ensure_tensor(obj) -> NDTensor: pass # given a tensor or JSON result, ensure that the object is NDTensor configured.
    ensure_data(obj) -> NDData: pass # given an JSON result, return a typed NDData wrapper if it isn't already.
    json_object(data:NDTensor|NDData) -> dict: pass # given a tensor return a JSON-stringify-able result.

    # Core API (NDTensor only):
    set_result(to :NDTensor): pass # set the destination tensor, and uses update semantics if provided. Returns new result if result is None
    push_pose(pose: NDTensor, unpose :NDTensor): pass # pushes a transform onto the stack (on the right), if pose is not provided, and unpose is provided, then the inverse of unpose will be pushed, otherwise it will be ignored.
    apply_data(data :NDTensor): pass # concatenates the data to existing input data given the current transform stack.
    pop_pose(): pass # pop the transform
    get_result() -> NDTensor: pass # returns the data transformed by the poses

    # NDObject Extensions:
    apply_children( obj :NDObject ): pass # walks the children calling pushPose/applyData/popPose as appropriate.
    update_from_children( obj :NDObject ): pass # updates the data on this node by walking it's child objects. Useful for scene caches.
    push_unpose_to_world( obj :NDObject, stopAt :NDObject=null): pass # walks the parents to preare this (camera?) to draw from world space
    update_from_world( obj :NDObject ): pass # updates the data on this node by pushing the unpose to world, and then walking the world. Useful for cameras.
```

## API - TypeScript + Three.js

```typescript
class NDSceneIn3D extends Object3D {
    objects : NDObject;

    patchScene( obj : NDScene|object );

    resultFromChildren( obj :NDObject ) :NDTensor;
    resultFromWorld( obj :ObjectN D) :NDTensor;
}
```

