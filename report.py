from reports.report_generator import ReportGenerator

if __name__ == '__main__':
    r = ReportGenerator(config='config/report_config.yaml')
    r.generate()
