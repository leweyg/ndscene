
# NDRender and NDScene (ndscene)
n-dimensional scene-graph runtime and format; allowing AI vision models to be expressed as the updating of an tensor-asset within a posed scene-graph. Such as a virtual camera or tile within a field of known camera views. Dictionaries of tensors are used throughout to provide flexability and named multi-dimensionality in addition to dense tensor formats.

## Abstract / Architectural Principals

* Scene Graph: there are three main uses of scene-graphs:
    * Managing collections of related elements: `node : { name:str, pose:tensor, data:tensor, children:[node] }`
    * Evaluating the sum of a node and it's children: `node.value() : node.pose * concat( node.data, c.value() for c in node.children )`
    * Updating the view from a 'camera/viewer' node: `node.update() : node.data = concat(c.unpose for c in node.parents_to_root()) * root.value()` 
* Render Kernel (NDRender): is an dictionary-of-tensor enhanced version of the original OpenGL1-style rendering API:
    * `ndBegin(target)`
    * `ndPush(pose,unpose)` # pushes a transform/inverse
    * `ndConcat(data)` # append this data transformed by current pose stack
    * `ndPop()` # pop an item from the transform stack
    * `ndEnd()` # return the 
* Tensors (target/transform/data) can be nestable dictionaries/lists of native/parameterized/autograd tensors or named methods/modules/dimensions (NDTensor|dict|list|torch.tensor).
    * `{ "vertices":{"shape":[8,3],"data":[1.0,0.0,1.0,...],`
    * `"indices":{"shape":[12,3],"dtype":"uint16","data":[0,2,1,...],`
    * `"colors":{"shape":[8,3],"dtype":"float","data":[0.5,1.0,0.5,...]}`
* Pose/Unpose (NDTensor) is used to pose data into it's parent space, or unpose it back into it's child/data space. I.e. you can transform filtered dictionaries of tensors using matrix multiplication (default), listed sequences of steps, or with an extensible set of standard transforms (append_ones, projection, index_to_). Inversion/"unpose" is used to support optimization and target relative transforms such as viewport encoding.
* Data/media (NDData) behind tensors can be progressivly transitioned between native-tensored, buffered, mime-compressed and remote-pathed states. Allowing natural integration of standard image, video, zip and other compression schemes.
* Streaming is achieved via scene patches/updates, including scenes which are themselves queries for additional content, and which generally leverage a secondary path-based file/shared-memory system for same-device or cacheable content.
* "Hardware Acceleration" (NDTorch) is achieved via PyTorch which is required for rendering (with CPU fallback), but optional for basic scene-graph file-IO.

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
                "compressed_type" : "string_mime_type|string_dtype",
                "buffer" : "bytes",
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

    # Core API (NDTensor only):
    ndBegin(to :NDTensor, per_input=True): pass # set the destination tensor, and uses update semantics if provided. Returns new result if result is None
    ngPush(pose: NDTensor, unpose :NDTensor): pass # pushes a transform onto the stack (on the right), if pose is not provided, and unpose is provided, then the inverse of unpose will be pushed, otherwise it will be ignored.
    ndConcat(data :NDTensor): pass # concatenates the data to existing input data given the current transform stack.
    ndPop(): pass # pop the transform
    ndEnd() -> NDTensor: pass # returns the data transformed by the poses

    # NDTensor/JSON conversions:
    native_tensor(obj) -> NDTensor: pass # given a tensor or JSON result, ensure that the object is NDTensor configured.
    ensure_data(obj) -> NDData: pass # given an JSON result, return a typed NDData wrapper if it isn't already.
    json_from(data:NDTensor|NDData) -> dict: pass # given a tensor return a JSON-stringify-able result.

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

