
# GLTF Details Viewer

This is a simple web app that opens a .glb/.gltf file by URL path (and defaults to door.gltf.glb), and displays a collapsible example tree view of as the data structures within. It tries to show as close as possible to the raw GLTF data heiarchy. Large buffers/array are shown as truncated lines (with size). Most data is a simple tree view. Items which are indices "INDEX" to other buffers are shown as "@IMPLIEDTYPE[INDEX]" rather than "INDEX".

