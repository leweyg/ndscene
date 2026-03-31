
# ndScene 



n-dimensional scene-graph runtime and format; allowing AI vision models to be expressed as the updating of an tensor-asset within a posed scene-graph. Such as a virtual camera or tile within a field of known camera views. Dictionaries of tensors are used throughout to provide flexability and named multi-dimensionality in addition to dense tensor formats.

## The N-Dimensional Engine Stack

* Streams of Packets: Generally with stream names, MIME format, byte buffers, usually decoded into nested tensors."
    * <code>class NDData : { path: str, format: str, buffer: bytes, <span style='color: blue;'>tensor: ndarray|pytorch.Tensor</span> }</code>
    * Basically fetch/file decoding.
* Heiarchial Tensor Structured Content: recursive tensors can be used to nest data like a JSON with native-tensors.
    *  <code>class NDTensor : { key: str, size: int, <span style='color: blue;'>shape: [NDTensor],</span> dtype: str, data: NDData }</code>
    * Recursive `shape` allows expression of data heiarchy, not possible with just <code><span style='color: red;'>shape:[int]</span></code> that native tensors use.
* Relationally-Transformed Objects: from spatial relationships to encode/decode networks as expressable as cacheable bi-directional graph from local to parent space:
    *  `class NDObject : { name:str, content:tensor, pose:tensor, unpose:tensor, parents:[NDObject], children:[NDObject] }`
    * Mathematically: <code><span style='color: blue;'>value = pose * content</span> | concat( pose * children, unpose * parents )</code>
    * Database: `CREATE TABLE NDObjectVersion ( version_id, object_id, scene_commit_id, state: NDObject, version_patch_parent? )`
* Scenes as Moments: A particular moment of objects, data caches, and method groups, often an version/instance on a timeline*
    * <code>class NDScene : { <span style='color: blue;'>root: NDObject</span>, objects:dict<str,NDObject>, tensors:<dict,NDTensor>, data:dict<str,NDData>, methods:dict<str,NDMethod>}</code>
    * `CREATE TABLE NDSceneCommit( scene_id, scene_commit_id, packet_data )` updates multiple objects.
* Updates as Model Inferences: Using a stack-style linearization of the graph walk we convert the updating of a scene element to an inference model format.*
    * `class NDRender:`
    * `ndAddModels(dict<str,callback>)`
    * <code><span style='color: blue;'>ndUpdateObjectInScene(obj:NDObject,scene:NDScene)</span></code>
    * `ndBegin(target)`
    * `ndPush(pose-encoder,unpose-decoder)` # push a transform or inverse-transform onto the stack
    * `ndConcat(data)` # append this data transformed by current pose stack
    * `ndPop()` # pop an item from the transform stack
    * `ndEnd()` # return the 
    * Returns a model which can then be run/trained to update the target from the posed input data.
* Labelled Model Data for Training*
    * `CREATE TABLE NDModelLabel ( Model:STR_ID, event_id:STR_ID, context: NDSceneCommit, input: NDSceneCommit, label: NDSceneCommit )`
    * Recorded model inputs and target layouts, along with modelling loss as the item being updates.
* Review Labelling Interface *
    * `NDEditor { view:NDSceneCommit, labels:range<NDModelLabel>, }` provides a visual review and editting/labelling interface to the database of commits (including UI state).*

## Abstract

* At the top level the two main activities of an engine are inferring scene-graph updates based on streamed inputs, and building a collection of reviewable and labelable training data.
* Pose/Unpose (NDTensor) is used to pose data into it's parent space, or unpose it back into it's child/data space. I.e. you can transform filtered dictionaries of tensors using matrix multiplication (default), listed sequences of steps, or with an extensible set of standard transforms (append_ones, projection, index_to_). Inversion/"unpose" is used to support optimization and target relative transforms such as viewport encoding.
* Data/media (NDData) behind tensors can be progressivly transitioned between remote-paths, buffers, and native-tensors. Allowing natural integration of standard image, video, zip and other compression schemes.
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

