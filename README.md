
# ndScene 



n-dimensional scene-graph runtime and format; allowing AI vision models to be expressed as the updating of an tensor-asset within a posed scene-graph. Such as a virtual camera or tile within a field of known camera views. Dictionaries of tensors are used throughout to provide flexability and named multi-dimensionality in addition to dense tensor formats.

## The N-Dimensional Engine Stack


| Name                                    | Description                                                                                                                                                     | Code Definition                                                                                                                                               | DB Representation                                                                                                                                                                                                                                                                                   |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Streams of Packets**                  | Raw byte/file/stream layer. Holds named packet data with format metadata and optionally decoded tensor content.                                                 | <code>class NDData: path: str, format: str, buffer: bytes, tensor: ndarray</code>                                                            | <code>NDDataBlob( data_id, path, format, buffer, tensor_cache_ref )</code>                                                                                                                                                                                                                          |
| **Tensor Structured Content**           | Hierarchical tensor-content layer. Represents nested tensor structures with recursive shape/content relationships beyond flat native tensor shapes.             | <code>class NDTensor: key: str, size: int, shape: list[NDTensor], dtype: str, data: NDData</code>                                                             | <code>NDTensorVersion( tensor_id, tensor_version_id, key, size, dtype, data_id, shape_patch_parent )</code><br><code>NDTensorShapeEdge( tensor_version_id, child_tensor_version_id, ordinal )</code>                                                                                                |
| **Object Transform-Graph** | Object/scene-graph layer. Represents content in local and parent-relative spaces using pose/unpose relationships across parent/child object links.              | <code>class NDObject: name: str, content: tensor, pose: tensor, unpose: tensor, parents: list[NDObject], children: list[NDObject]</code>                      | <code>NDObjectVersion( version_id, object_id, scene_commit_id, state, version_patch_parent )</code><br><code>NDObjectEdge( object_version_id, related_object_version_id, relation_type, ordinal )</code>                                                                                            |
| **Packets as Scenes Commits**           | Scene snapshot / commit layer. Collects objects, tensors, data, and methods into a versioned scene state that can be updated by internal or external packets.   | <code>class NDScene: root: NDObject, objects: dict[str, NDObject], tensors: dict[str, NDTensor], data: dict[str, NDData], methods: dict[str, NDMethod]</code> | <code>NDSceneCommit( scene_id, scene_commit_id, packet_data, is_external )</code><br><code>NDSceneCommitObject( scene_commit_id, object_version_id )</code><br><code>NDSceneCommitTensor( scene_commit_id, tensor_version_id )</code><br><code>NDSceneCommitData( scene_commit_id, data_id )</code> |
| **Updates as Model Inferences**         | Inference/update execution layer. Linearizes graph-relative scene updates into a stack-based model input/output process for rendering, prediction, or training. | <code>class NDRender: ndTarget(obj) ndPush(pose,unpose), ndConcat(content), ndPop(), ndTargetModel(), ndTargetUpdate() </code>                                                                                                                                   | <code>NDInferenceEvent( event_id, model_id, target_object_id, scene_commit_id, input_commit_id, output_commit_id )</code><br><code>NDInferenceTrace( event_id, op_index, op_type, op_data )</code>                                                                                                  |
| **Labelled Model Data for Training**    | Supervised training record layer. Stores model events with context, input, and target scene commits for learning and replay.                                    | <code>class NDModelLabel: Model: STR_ID, event_id: STR_ID, context: NDSceneCommit, input: NDSceneCommit, label: NDSceneCommit</code>                          | <code>NDModelLabel( model_id, event_id, context_commit_id, input_commit_id, label_commit_id )</code>                                                                                                                                                                                                |
| **Review Labelling Interface**          | Human review and editing layer. Provides commit inspection and label editing over recorded model/training data.                                                 | <code>class NDEditor: view_commits: NDSceneCommit, labels: range[NDModelLabel]</code>                                                                         | <code>NDEditSession( session_id, user_id, view_commit_id, label_range_start, label_range_end, ui_state )</code>                                                                                                                                                                                     |


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

