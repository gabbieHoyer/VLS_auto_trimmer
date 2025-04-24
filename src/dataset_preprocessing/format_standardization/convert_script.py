import os
import xml.etree.ElementTree as ET

def convert_voc_to_txt(xml_dir, output_dir):
    """
    Convert Pascal VOC XML annotations to TXT files with [x_min, y_min, width, height].
    
    Args:
        xml_dir (str): Directory containing XML annotation files.
        output_dir (str): Directory to save converted TXT files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            txt_file = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
            with open(txt_file, 'w') as f:
                for obj in root.findall('object'):
                    if obj.find('name').text == 'person':
                        bbox = obj.find('bndbox')
                        x_min = float(bbox.find('xmin').text)
                        y_min = float(bbox.find('ymin').text)
                        x_max = float(bbox.find('xmax').text)
                        y_max = float(bbox.find('ymax').text)
                        width = x_max - x_min
                        height = y_max - y_min
                        f.write(f"{x_min} {y_min} {width} {height}\n")

if __name__ == "__main__":
    # Define input and output directories
    xml_dir = "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_labels_part4"
    output_dir = "/data/mskscratch/users/ghoyer/Precision_Air/0403/face_videos/frame_labels_txt_part4"
    
    print(f"Converting XML annotations from {xml_dir} to TXT files in {output_dir}")
    convert_voc_to_txt(xml_dir, output_dir)
    print("Conversion complete.")

# python -m src.dataset_preprocessing.format_standardization.convert_script

# Annotation Format
# The annotations were originally in Pascal VOC XML format (output by LabelImg) with the label person.
# The XML files were converted to .txt files using the script convert_voc_to_txt, which:
# Reads each XML file.
# Extracts bounding boxes for objects labeled as person.
# Converts the bounding box format from [xmin, ymin, xmax, ymax] to [x_min, y_min, width, height].
# Writes each bounding box to a .txt file, one per line, as space-separated values: x_min y_min width height.