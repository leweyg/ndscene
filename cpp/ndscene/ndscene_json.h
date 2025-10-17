
// Simplistic JSON parser for ndscene files.

#include <memory>
#include <string>
#include <vector>
#include <stdio.h>

#include "ndtensor.h"

class NdJsonParser {
protected:
    const char* mText = nullptr;
    size_t mPos = 0;
    size_t mLen = 0;
    NdTensorPtr mResult;
    NdTensorPtr mDebugLatest;

    NdTensorPtr parseObject() {
        skipWhitespace();
        NdTensorPtr ans = NdTensor::MakeShared();
        char letter = currentLetter();
        if (letter == '\"') {
            std::string key = parseQuoted();
            ans->mData = NdData::MakeShared();
            ans->mData->setText(key.c_str());
            return ans;
        } else if (isNumberChar(letter)) {
            auto start = mPos;
            while (isNumberChar(letter) && stepNext()) {
            }
            auto end = mPos;
            std::string numberStr(mText + start, end - start);
            ans->mData = NdData::MakeString(numberStr.c_str());
            skipWhitespace();
            return ans;
        } else if (letter == '[') {
            stepNext(); skipWhitespace();
            while (currentLetter() != ']') {
                auto item = parseObject();
                ans->mShape.push_back(item);
                if (currentLetter() == ',') {
                    stepNext(); skipWhitespace();
                }
            }
            stepNext(); skipWhitespace();
            return ans;
        } else if (letter == '{') {
            stepNext();
            skipWhitespace();
            auto prevPos = ~mPos;
            while (currentLetter() != '}') {
                if (mPos == prevPos) {
                    printCurrent();
                }
                ND_ASSERT(mPos != prevPos);
                prevPos = mPos;
                if (currentLetter() == '\"') {
                    skipWhitespace();
                    std::string key = parseQuoted();
                    if (currentLetter() == ':') {
                        stepNext();
                        NdTensorPtr val = parseObject();
                        val->mKey = key;
                        ans->mShape.push_back(val);
                    }
                    continue;
                }
                if (currentLetter() == ',') {
                    stepNext(); skipWhitespace();
                    continue;
                }
                if (currentLetter() == '}') {
                    stepNext(); skipWhitespace();
                    return ans;
                }
                if (mPos == prevPos) {
                    printCurrent(ans);
                }
                ND_ASSERT(mPos != prevPos);
            }
        } else {
            printCurrent(ans);
            ND_ASSERT(false);
        }
        return ans;
    }

    void printCurrent(NdTensorPtr ptr = nullptr) {
        int line = 1;
        int col = 1;
        for (size_t i=0; i<mPos; i++) {
            if (mText[i] == '\n') {
                line++;
                col=1;
            } else {
                col++;
            }
        }
        std::string clip;
        const int max_clip_len = 6;
        for (int i=0; i<max_clip_len; i++) {
            if ((mPos+i) < mLen) {
                clip += mText[mPos+i];
            }
        }
        printf("Current letter: '%c' of '%s' line %d, col %d, pos %d \n", currentLetter(), clip.c_str(), line, col, (int)mPos);
        if (ptr) {
            printf("Current state: %s\n", ptr->asString().c_str());
        }
    }

    std::string parseQuoted() {
        ND_ASSERT(currentLetter() == '\"');
        stepNext();
        size_t start = mPos;
        while (currentLetter() != '\"' && stepNext()) {
        }
        size_t end = mPos;
        ND_ASSERT(currentLetter() == '\"');
        stepNext();
        skipWhitespace();
        std::string ans(mText + start, end - start);
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

    static bool isNumberChar(char letter) {
        if (letter >= '0' && letter <= '9') {
            return true;
        }
        if (letter == '.') {
            return true;
        }
        if (letter == '-') {
            return true;
        }
        return false;
    }

    void skipWhitespace() {
        while (isWhiteSpace(currentLetter())) {
            if (!stepNext()) {
                return;
            }
        }
    }

    char currentLetter() const {
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

    bool isAtEnd() const {
        return ((mPos + 1) >= mLen);
    }

public:
    NdJsonParser(const char* pText) { 
        mText = pText; 
        mPos = 0;
        mLen = strlen(pText);
    }

    NdTensorPtr parse() {
        if (mResult) {
            return mResult;
        }
        mResult = parseObject();
        return mResult;
    }

};

