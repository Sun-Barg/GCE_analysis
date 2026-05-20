import xml.etree.ElementTree as ET

xml_file = 'XML_models_12yr/GCE_12yr_4FGLDR2_Model_I.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

free_count = 0
total_count = 0

for source in root.findall('source'):
    for param in source.iter('parameter'):
        total_count += 1
        if param.get('free') == '1' or param.get('free') == 'true':
            free_count += 1

print(f"📊 총 파라미터 개수: {total_count}개")
print(f"🔥 현재 열려있는(Free) 파라미터 개수: {free_count}개")
