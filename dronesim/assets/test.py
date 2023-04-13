import xml.etree.ElementTree as ET
tree = ET.parse('Falcon_debug.urdf')
root = tree.getroot()

mass = root.find("link/inertial/mass")
M = float(mass.attrib['value'])
print(M)

mass.attrib['value'] = str(10.4)
M = float(mass.attrib['value'])
print(M)

tree.write('Falcon_debug.urdf')