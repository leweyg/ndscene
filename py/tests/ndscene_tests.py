
print("Importing...")

# without install:
import ndscenepy.ndscene as ndscene

#import ndscene

print("Defining...")

def main_tests():
    print("ndscene_tests.main_tests starting:")
    desc = ndscene.ndobject('just a test')
    print("desc=", desc)
    print("ndscene_tests.main_tests done.")
    pass;

if __name__ == "__main__":
    main_tests()
