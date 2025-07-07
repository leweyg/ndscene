
# SceneND: ObjectND, TensorND and RenderND
n-dimensional scene-graph format and runtime

## JSON Schema

In brief: scenes contain objects which contain dictionaries of tensors. Object transforms can be selection from shared tensors. The @type semantics below are optional and included for clarity.

```json
{
    // 
    "@type": "SceneND",
    "objects" :{"key :string_key":{
        "@type": "ObjectND",
        "children": "list<string_key|ObjectND",)
        "parents": "list<string_key|ObjectND",
        "pose": "TensorND", // read(): return pose * concat( data, children.read() )
        "unpose": "TensorND", // write(value): data = unpose * value
        "data": {

            "@type" :"TensorND",
            "key" :"string", // the dimension name / dictionary key
            "size" :"int", // the size used in 'shape:[int]' shorthand
            "shape" :"list<TensorND>", // nested for dictionaries
            "dtype" :"string", // data element type, traditionally an enum
            "data" : {
                
                "@type" : "DataND |array|bytes|string", // shorthand types
                "path": "string", // path to shared memory, runtime dependant
                "array" : ["number|string|etc."], // primary array
                // custom types (for effecient typed transport):
                "text" : "string",
                "buffer" : "bytes",
                "strings": ["string"],
                "numbers": ["number"],
            }
        }
    }}
}
```

For convenience these can be specified in pure JSON, and don't need to be strongly typed before being passed to the Render APIs. Unrecognized dictionary keys are considered shape properties (in the provided order), and shapes without named keys are considered sizes:

| JSON | TensorND |
| --- | --- |
| `{shape:[2,3],data=[1,2,3,4,5,6]}` | `{shape:[{size:2},{size:3}],data:[1,2,3,4,5,6],dtype=number,size:6}` |
| `{foo:bar,etc:8}` | `{shape:[{key:foo,data:bar},{key:etc,data:8}]}` |


## API - Python

```python
class RenderND:

    # TensorND/JSON conversions:
    ensure_tensor(obj) -> TensorND: pass # given a tensor or JSON result, ensure that the object is TensorND configured.
    ensure_data(obj) -> DataND: pass # given an JSON result, return a typed DataND wrapper if it isn't already.
    json_object(data:TensorND|DataND) -> dict: pass # given a tensor return a JSON-stringify-able result.

    # Core API (TensorND only):
    set_result(to :TensorND): pass # set the destination tensor, and uses update semantics if provided. Returns new result if result is None
    push_pose(pose: TensorND, unpose :TensorND): pass # pushes a transform onto the stack (on the right), if pose is not provided, and unpose is provided, then the inverse of unpose will be pushed, otherwise it will be ignored.
    apply_data(data :TensorND): pass # concatenates the data to existing input data given the current transform stack.
    pop_pose(): pass # pop the transform
    get_result() -> TensorND: pass # returns the data transformed by the poses

    # ObjectND Extensions:
    apply_children( obj :ObjectND ): pass # walks the children calling pushPose/applyData/popPose as appropriate.
    update_from_children( obj :ObjectND ): pass # updates the data on this node by walking it's child objects. Useful for scene caches.
    push_unpose_to_world( obj :ObjectND, stopAt :ObjectND=null): pass # walks the parents to preare this (camera?) to draw from world space
    update_from_world( obj :ObjectND ): pass # updates the data on this node by pushing the unpose to world, and then walking the world. Useful for cameras.
```

## API - TypeScript + Three.js

```typescript
class SceneNDIn3D extends Object3D {
    objects : ObjectND;

    patchScene( obj : SceneND|object );

    resultFromChildren( obj :ObjectND ) :TensorND;
    resultFromWorld( obj :ObjectN D) :TensorND;
}
```

