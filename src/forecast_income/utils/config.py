import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(path: str = "config/params.yaml") -> Dict[str, Any]:
    """
    Carga la configuración y fusiona los parámetros específicos del modo (daily/monthly)
    en las secciones principales.
    """
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    
    mode = params.get("mode", "monthly")
    
    # Merge mode specific params into main sections
    if "modes" in params and mode in params["modes"]:
        mode_params = params["modes"][mode]
        
        # Update top level sections
        for section, values in mode_params.items():
            if isinstance(values, dict) and section in params and isinstance(params[section], dict):
                params[section].update(values)
            else:
                params[section] = values
                
    params["active_mode"] = mode
    return params
