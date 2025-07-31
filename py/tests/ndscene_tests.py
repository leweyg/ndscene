
print("Importing...")

# without install:
import ndscenepy.ndscene as ndscene

#import ndscene

print("Defining...")

def main_test_tensors():
    array = [ [ 1, 2, 3 ], [ 4, 5, 6 ] ]
    tnd = ndscene.NDTensor.from_arrays(array)
    tnr = ndscene.NDTorch.tensor(tnd)
    print(tnr)

def main_tests():
    print("ndscene_tests.main_tests starting:")
    desc = ndscene.NDObject()
    print("desc=", desc)
    print("ndscene_tests.main_test_tensors:")
    main_test_tensors()
    print("ndscene_tests.main_tests done.")
    pass;

if __name__ == "__main__":
    main_tests()
