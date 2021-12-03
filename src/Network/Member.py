class Member:
    """
    TODO Docstring Member
    """

    def __init__(self, properties: dict):
        """
        TODO Docstring Member __init__
        """
        self.properties = {}
        for property, value in properties.items():
            self.properties[property] = value

    def __str__(self):
        return str(self.properties)
