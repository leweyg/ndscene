
print("Importing...")

# without install:
import ndscenepy.ndscene as ndscene

#import ndscene

print("Defining...")

def text_layout_example():
    text = "run layout\ntest on this."
    text_tensor = ndscene.JsonND.ensure_tensor(text)

def main_example():
    print("ndscene_example starting:")
    text_layout_example()
    print("ndscene_example done.")
    pass

if __name__ == "__main__":
    main_example()
