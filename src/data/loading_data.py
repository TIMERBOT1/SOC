import json


class LoadingData:
    text_file_path = 'text_content_SOC.txt'
    table_file_path = 'latex_tables_SOC_v1.txt'
    tables = None
    text_elements = None

    def get_tables(self):
        if self.tables is None:
            with open(self.table_file_path) as json_file:
                self.tables = json.load(json_file)
        return self.tables

    def get_text_element(self):
        if self.text_elements is None:
            with open(self.text_file_path, 'r') as fr:
                self.text_elements = json.load(fr)
        return self.text_elements
