
# ObjectND, TensorND and RenderND
n-dimensional scene-graph format and runtime

## JSON Schema

In brief: scenes contain objects which contain dictionaries of tensors. Object transforms can be selection from shared tensors. The @type semantics below are optional and included for clarity.

```json
{
    "@type": "SceneND",
    "objects" :{"key :string_key":{

        "@type": "ObjectND",
        "children": "list<string_key|ObjectND",
        "parents": "list<string_key|ObjectND",
        "pose": "tensorND",
        "unpose": "tensorND",
        "data": {

            "@type" :"TensorND",
            "key" :"string",
            "size" :"int",
            "shape" :"list<TensorND>",
            "data" :"array_of_dtype",
            "dtype" :"string",
        }
    }}
}
```

For convenience these can be specified in pure JSON. Unrecognized dictionary keys are considered shape properties (in the provided order), and shapes without named keys are considered sizes:

| JSON | TensorND |
| --- | --- |
| `{shape:[2,3],data=[1,2,3,4,5,6]}` | `{shape:[{size:2},{size:3}],data:[1,2,3,4,5,6],dtype=number,size:6}` |
| `{foo:bar,etc:8}` | `{shape:[{key:foo,data:bar},{key:etc,data:8}]}` |


## API - Python

```python
class RenderND:
    # Core API (TensorND only):
    setResult(to :TensorND): pass # set the destination tensor, and uses update semantics if provided. Returns new result if not provided
    pushPose(pose: TensorND, unpose :TensorND): pass # pushes a transform onto the stack (on the right), if pose is not provided, and unpose is provided, then the inverse of unpose will be pushed, otherwise it will be ignored.
    applyData(data :TensorND): pass # concatenates the data to existing input data given the current transform stack.
    popPose(): pass # pop the transform
    getResult() -> TensorND: pass # returns the data transformed by the poses

    # ObjectND Extensions:
    applyChildren( obj :ObjectND ): pass # walks the children calling pushPose/applyData/popPose as appropriate.
    resultFromChildren( obj :ObjectND ): pass # updates the data on this node by walking it's child objects. Useful for scene caches.
    pushUnposeToWorld( obj :ObjectND, stopAt :ObjectND=null): pass # walks the parents to preare this (camera?) to draw from world space
    resultFromWorld( obj :ObjectND ): pass # updates the data on this node by pushing the unpose to world, and then walking the world. Useful for cameras.
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

### JSON Format and Previous Work


| ObjectND | Previous Work - Scene Graphs | Description |
| --- | --- | --- |
| key :string | name: string | Unique identifier |
| pose :TensorND | Pose : float4x4 | Value-space from local data/children-space |
| unpose :TensorND | Projection : float4x4 | Local data/children-space from value-space |
| data :TensorND | Components | Nestable dictionary/lists of tensors |
| parents :[ObjectND/string_id] | Parent :Object3D (singular) | Known relative transforms |
| children:[ObjectND/string_id] | Children: [ObjectND] | Sub-objects, value = concat(data,children) |

| TensorND | Previous Work - Tensors | Descriptions |
| --- | --- | --- |
| key :string | parent.names[i] | Dimension name (optional) |
| size :int | parent.shape[i] | Dimension size |
| data :[dtype] | data :[dtype] | Linear array of data |
| dtype:string | dtype:lib.type | String name of data element type |


| RenderND | Previous Work - Graphics | Description |
| --- | --- | --- |
| glndPushTarget(tensor) | glRenderTarget(target) | Output tensor |
| glndPushUnpose(tensor) | glCamera() | Camera configuration |
| glndPushPose(tensor) | glPushMatrix(f4x4) | Object transforms |
| glndValue(tensor) | glDraw() | Push value |
| glndPopPose() | glPopMatrix() | Pop object transform |
| glndPopUnpose() | glPopCamera() | Prop camera  |


`value() = pose * concat(data,children.value())`

`cache(): data = unpose * children.value()`

`draw(): data = unpose * world * pose * world.value()`