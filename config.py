class Config:
    version = 'base'
    
    base = True
    clip = False
    
    epochs = 10
    batch_size = 16
    
    
def get_default_config():
    return Config()


def get_clip_config():
    config = Config()
    config.version = 'clip'
    config.base = False
    config.clip = True
    return config