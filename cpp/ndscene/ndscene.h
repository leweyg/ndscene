
#include <memory>
#include <string>
#include <vector>
#include <map>

#include "ndtensor.h"

class NdObject {
public:
    std::string mName;
    NdTensorList mParents;
    NdTensorList mChildren;
    NdTensorPtr mContent;
    NdTensorPtr mPose;
    NdTensorPtr mUnpose;
};
typedef std::shared_ptr<NdObject> NdObjectPtr;

class NdScene {
    NdObjectPtr mRoot;
    std::map<std::string, NdObjectPtr> mObjects;
    std::map<std::string, NdTensorPtr> mTensors;
    //std::map<std::string, NdMethodPtr> mMethods;
    std::string mPathRoot;
};
