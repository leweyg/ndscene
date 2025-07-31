
print("Importing...")

# without install:
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
        glyph = text_tensor[i]
        if (glyph == '\n'):
            row = row + 1
            col = 0
            #continue
        elif (glyph == '\t'):
            spaces = 4
            col = ( col + spaces ) - ( col % spaces )
            #continue
        else:
            col = col + 1
        state[i,0] = col - 1
        state[i,1] = row
        state[i,2] = ord(glyph)
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
    text = "This is\na\ttest."
    text_data = ndscene.DataND.from_text(text)
    text_tensor = text_data.ensure_tensor()
    print("text.shape=", text_tensor.shape)

    layout = layout_text(text_tensor)

    console = torch.zeros( [2, 10], dtype=torch.uint8 )
    console = render_console( console, layout )
    #print(console)
    out_text = text_from_console(console)
    print(out_text)

    return text_data

def main_tests():
    print("ndscene glyph test starting:")
    res = scene_of_text()
    print("res=", res)
    print("ndscene glyph test done.")
    #exit(1)
    pass;

if __name__ == "__main__":
    main_tests()
