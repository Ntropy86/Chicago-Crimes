class ChicagoCrimeException(Exception):
    """Base Exception Class"""
    pass
class DatasetDownloadError(Exception):
    """Error class for when theres an issue in Downloading the raw dataset"""
    pass
class DataProcessingError(Exception):
    """Error for Processing the Data"""
    pass
class ConfigError(Exception):
    """Config Error"""
    pass

