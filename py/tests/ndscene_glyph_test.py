
print("Importing...")

# without install:
import ndscenepy.ndscene as ndscene

#import ndscene

print("Defining...")

def scene_of_text():
    text = "This is\na test."
    text_data = ndscene.DataND.from_text(text)
    ans = text_data.ensure_tensor()
    print("text.shape=", ans.shape)

    return ans

def main_tests():
    print("ndscene glyph test starting:")
    desc = scene_of_text()
    print("desc=", desc)
    print("ndscene glyph test done.")
    #exit(1)
    pass;

if __name__ == "__main__":
    main_tests()
