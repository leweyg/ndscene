
#include <stdio.h>
#include <string>

#include "../ndscene/ndscene_json.h"

#include "test.h"

const char* gTextFilePath = "json/freed_go/voxels.json"; //"json/freed_go/view_1_scene.json";

std::string readFileAsString(const char* path) {
    FILE* fp = fopen(path, "r");
    ND_ASSERT(fp);
    if (fseek(fp, 0L, SEEK_END) != 0) {
        // Handle error: fseek failed
        fclose(fp);
    }
    size_t file_size = ftell(fp);
    fseek(fp, 0L, 0);
    char* buffer = (char*)malloc(file_size + 1);
    fread(buffer, file_size, 1, fp);
    buffer[file_size] = 0;
    std::string ans = std::string(buffer);
    free(buffer);
    fclose(fp);
    return ans;
}

int main(int argc, char** argv) {
    printf("Hello, World!\n");

    SortingTest::RunTest();
    return 0;

    std::string jsonText = readFileAsString(gTextFilePath);
    printf("JsonText='%s'\n", jsonText.c_str());

    auto node = NdJsonParser(jsonText.c_str()).parse();
    printf("Node=%s\n", node->asString().c_str());

    return 0;
}