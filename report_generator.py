import os
from jinja2 import Environment, FileSystemLoader


class ReportGenerator:
    """
    使用Jinja2模板和Plotly图表生成HTML报告。
    """

    def __init__(self, report_path, template_str):
        self.report_path = report_path
        # 使用字符串作为模板，避免需要额外的模板文件
        self.env = Environment(loader=FileSystemLoader('.'))
        self.template = self.env.from_string(template_str)

    def generate_report(self, filename, context):
        """
        渲染模板并保存HTML文件。
        """
        # 将Plotly图表转换为HTML div
        for key, fig in context['figures'].items():
            context['figures'][key] = fig.to_html(full_html=False, include_plotlyjs='cdn')

        html_content = self.template.render(context)

        filepath = os.path.join(self.report_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"网页报告已成功生成: {filepath}")
