class Member:
    def __init__(self, properties: dict):
        self.properties = {}
        for property, value in properties.items():
            self.properties[property] = value

    def __str__(self):
        return str(self.properties)
