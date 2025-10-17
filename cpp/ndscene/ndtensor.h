
#include <memory>
#include <string>
#include <vector>


#define ND_ASSERT(stmt) {if(!(stmt)){NdUtils::FailedAssert(#stmt, __FILE__, __LINE__);}}

class NdUtils {
public:
    static void FailedAssert(const char* stmt, const char* file, int line) {
        printf("Failed Assert:%s at %s(%d)\n", stmt, file, line);
        exit(1);
    }
};


class NdDataType {
public:
    std::string mTypeName;
};

class NdData {
public:
    std::string mText;

    NdData() {}
    NdData(const char* text) : mText(text) {
    }

    void setText(const char* text) {
        mText = text;
    }

    static std::shared_ptr<NdData> MakeShared() {
        return std::make_shared<NdData>();
    }
    static std::shared_ptr<NdData> MakeString(const char* str) {
        return std::make_shared<NdData>(str);
    }
};

class NdTensor {
public:
    std::string mKey;
    int mSize = 0;
    std::vector<std::shared_ptr<NdTensor>> mShape;
    std::shared_ptr<NdDataType> mDType;
    std::shared_ptr<NdData> mData;

    NdTensor() {}

    static std::shared_ptr<NdTensor> MakeShared() {
        return std::make_shared<NdTensor>();
    }

    std::string asString() {
        std::string ans("{");
        if (mKey.size()) {
            ans += "'";
            ans += mKey.c_str();
            ans += "':";
        }
        if (mShape.size()) {
            ans += "[";
            for (auto s : mShape) {
                ans += s->asString();
                ans += ",";
            }
            ans += "]";
        }
        if (mData) {
            ans += "=";
            ans += mData->mText;
        }
        if (mDType) {
            ans += "<";
            ans += mDType->mTypeName;
            ans += ">";
        }
        ans += "}";
        return ans;
    }
};
typedef std::shared_ptr<NdTensor> NdTensorPtr;
typedef std::vector<NdTensorPtr> NdTensorList;
