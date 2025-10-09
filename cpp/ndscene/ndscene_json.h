
// Simplistic JSON parser for ndscene files.

#include <memory>
#include <string>
#include <vector>
#include <stdio.h>

#define ND_ASSERT(stmt) {if(!(stmt)){NdUtils::FailedAssert(#stmt, __FILE__, __LINE__);}}

class NdUtils {
public:
    static void FailedAssert(const char* stmt, const char* file, int line) {
        printf("Failed Assert:%s at %s(%d)\n", stmt, file, line);
    }
};

class NdJsonNode {
protected:
    std::string mKey;
    std::string mValue;
    std::vector<std::shared_ptr<NdJsonNode>> mChildren;
public:
    NdJsonNode() {}
};
typedef std::shared_ptr<NdJsonNode> NdJsonNodePtr;

class NdJsonParser {
protected:
    const char* mText = nullptr;
    size_t mPos = 0;
    size_t mLen = 0;
    NdJsonNodePtr mResult;

    NdJsonNodePtr parseObject() {
        skipWhitespace();
        NdJsonNodePtr ans = std::make_shared<NdJsonNode>();
        return ans;
    }

    static bool isWhiteSpace(char letter) {
        switch (letter) {
            case ' ':
            case '\t':
            case '\n':
            case '\v':
                return true;
            default:
                return false;
        }
    }

    void skipWhitespace() {
        while (isWhiteSpace(currentLetter())) {
            stepNext();
        }
    }

    char currentLetter() {
        return mText[mPos];
    }

    bool stepNext() {
        if ((mPos+1) < mLen) {
            mPos++;
            return true;
        } else {
            return false;
        }
    }

public:
    NdJsonParser(const char* pText) { 
        mText = pText; 
        mPos = 0;
        mLen = strlen(pText);
    }

    NdJsonNodePtr parse() {
        if (mResult) {
            return mResult;
        }
        mResult = parseObject();
        return mResult;
    }

};

