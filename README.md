
# ObjectND, TensorND and RenderND
## n-dimensional scene-graph format and runtime

### Format and Previous Work


| ObjectND | Previous Work - Scene Graphs | Description |
| --- | --- | --- |
| pose :TensorND | Pose : float4x4 | Data/children to local transform/encoding |
| unpose :TensorND | Projection : float4x4 | Local to data transform/encoding. Defaults to inverse of pose. |
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