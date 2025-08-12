
print("Importing...")

# without install:
import os
print("pwd=", os.getcwd())
import sys
sys.path.append(os.getcwd())

import ndscenepy.ndscene as ndscene
import torch

#import ndscene

print("Defining...")

def layout_text(text_tensor):
    count = text_tensor.shape[0]
    semi_struct = {
        "col":0,
        "row":1,
        "glyph":2
    }
    state = torch.zeros( [count,len(semi_struct)], dtype=torch.int32 )
    col = 0
    row = 0
    glyph = 0
    for i in range(count):
        glyph = text_tensor[i] #.item()
        if (glyph == ord('\n')):
            row = row + 1
            col = 0
            #continue
        elif (glyph == ord('\t')):
            spaces = 4
            col = ( col + spaces ) - ( col % spaces )
            #continue
        else:
            col = col + 1
        state[i,0] = col - 1
        state[i,1] = row
        state[i,2] = glyph # ord(glyph)
    #print(state)
    return state

def render_console(dst, layout):
    count = layout.shape[0]
    for i in range(count):
        x = layout[i,0]
        y = layout[i,1]
        g = layout[i,2]
        dst[y,x] = g
    return dst

def text_from_console(dst):
    ans = "";
    for y in range(dst.shape[0]):
        for x in range(dst.shape[1]):
            glyph = dst[y,x]
            if (glyph == 0):
                glyph = ' '
            else:
                glyph = chr(glyph)
                if (glyph == '\n'):
                    glyph = ' '
                if (glyph == '\t'):
                    glyph = ' '
            ans += glyph
        ans += "\n"
    return ans

def scene_of_text():
    scene = ndscene.NDScene()

    text = "This is\na\ttest."
    text_data = ndscene.NDData.from_text(text)
    text_tensor = ndscene.NDTensor.from_data(text_data)
    text_node = ndscene.NDObject(content=text_tensor, scene=scene)
    scene.add_tensor("text_data", text_tensor)
    text_tensor = text_data.native_tensor(scene)
    print("text.shape=", text_tensor.shape)

    layout_data = layout_text(text_tensor)
    layout_node = ndscene.NDObject(content=layout_data)
    layout_node.pose = "layout_text"
    #root = layout_node.child_add(root)
    layout_node.child_add(text_node)
    scene.root = layout_node
    

    console = torch.zeros( [2, 10], dtype=torch.uint8 )
    console = render_console( console, layout_data )
    #print(console)
    out_text = text_from_console(console)
    print(out_text)

    return None # scene

def main_tests():
    print("ndscene glyph test starting:")
    res = scene_of_text()
    if res:
        print("res=", res)
    print("ndscene glyph test done.")
    #exit(1)
    pass;

if __name__ == "__main__":
    main_tests()
