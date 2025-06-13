


class ndjson:
    def str_property(property_name:str, property_value:str) -> str:
        # TODO(leweyg): string validation on both
        return f"\"{property_name}\":\"{property_value}\""
    def str_object1(property_name:str, property_value:str) -> str:
        inner = ndjson.str_property(property_name, property_value)
        return "{" + inner + "}";


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

    def __init__(self, initVal : str):
        self.text = initVal

    def __str__(self):
        if (self.text):
            return ndjson.str_object1('text', self.text)
        return "{}"


"""n-dimensional scene graph element, with graphics functions"""
class ndobject():

    size : int = 0
    """Size/length of this dimension"""

    name : str = None
    """Name or key of this dimension"""

    decode : "ndobject" = None
    """Transform from local/child space to parent space"""

    shape : list["ndobject"] = []
    """Children and/or tensor shape"""

    encode : "ndobject" = None
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
