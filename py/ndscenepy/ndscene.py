

class ndjson:
    def str_property(property_name:str, property_value:str) -> str:
        # TODO(leweyg): string validation on both
        return f"\"{property_name}\":\"{property_value}\""
    def str_object1(property_name:str, property_value:str) -> str:
        inner = ndjson.str_property(property_name, property_value)
        return "{" + inner + "}";

def nd_todo():
    raise "nd_todo"

class ndscenedata():
    path : str = None
    """Externally shared 'path' of the data"""

    text : str = None
    """Single text block"""

    numbers : list[float] = None
    """Array of numbers"""

    strings : list[str] = None
    """Array of strings"""

    buffer : bytes = None
    """Byte buffer"""

    def __init__(self, initVal):
        if (initVal is str):
            self.text = initVal
            return
        if (initVal is bytes):
            self.buffer = initVal
            return
        if (initVal is list):
            if (len(initVal) > 0):
                first = initVal[0]
                if (first is str):
                    self.strings = initVal
                    return
                elif (first is float):
                    self.numbers = initVal
                    return
                raise "OtherListType?=" + type(first)
            # empty array
            self.numbers = []
            return;

    def __str__(self):
        if (self.text):
            return ndjson.str_object1('text', self.text)
        return "{}"


"""n-dimensional scene graph element, with graphics functions
get value = pose * ((data<dtype> as shape) | shape )
set data by (value) = ((unpose | inv(pose)) * value) as shape)<dtype>
"""
class ndobject():

    size : int = 0
    """Size/length of this dimension"""

    name : str = None
    """Name or key of this dimension"""

    pose : "ndobject" = None
    """Transform from local/child space to parent space"""

    shape : list["ndobject"] = []
    """Children and/or tensor shape"""

    unpose : "ndobject" = None
    """Transform to data/child space from local space"""

    dtype : str = None
    """Data type (format) for this data"""

    data : ndscenedata = None
    """Data at this node"""

    def __init__(self, initVal : str|None):
        if (initVal):
            self.data = ndscenedata( initVal )

    def __str__(self):
        ans = "{"; #\"ndobject\":true,"
        if (self.size != 0):
            ans += "\"size\"=0,"
        if (self.name):
            ans += f"\"name\":\"{self.name}\","
        if (self.data):
            ans += f"\"data\":{self.data},";
        ans += "}"
        return ans

    @staticmethod
    def from_json(self, obj):
        if (obj is str):
            ans = ndobject()
            ans.data = ndscenedata(obj)
            return ans
        if (obj is dict):
            ans = ndobject()
            ans.shape = []
            for k,v in obj.items():
                d = ndobject.from_json(v)
                if (k and not d.name):
                    d.name = k
                ans.shape.append(d)
            return ans
        if (obj is list):
            ans = ndobject()
            ans.shape = []
            for v in obj:
                d = ndobject.from_json(v)
                ans.shape.append(d)
            return ans
        nd_todo()

    def json_object(self):
        if (self.pose):
            return todo_object
        if (self.data and self.shape):
            return todo_tensor
        if (self.data):
            return todo_array
        if (self.shape):
            return todo_dict
        if (self.name):
            return todo_str
        if (self.size):
            return todo_number
        

class ndscene():

    world : ndobject = None

    def redraw(self, target : ndobject) -> ndobject:
        return target
    
    def to_json_obj(self) -> dict:
        raise "TODO: ndscene.to_json_obj"
        return {}
    
    def to_json_str(self) -> str:
        obj = self.to_json_obj()
        import json;
        return json.dumps(obj)
