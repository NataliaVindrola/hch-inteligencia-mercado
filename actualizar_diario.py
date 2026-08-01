"""
Actualización diaria — Inteligencia de Mercado HCH

Ejecuta, en orden:
  1. actualizacion_diaria.ipynb   (TRM, CME, BMC, RONI, sacrificio bovino/porcino)
  2. etl_modelo_dimensional.ipynb (reconstruye el esquema estrella + vista mensual)

Al terminar, mercado_hch.db queda actualizado y Streamlit lo recoge solo
en la próxima vez que alguien abra el dashboard (la caché se invalida
automáticamente por fecha de modificación del archivo — no hace falta
reiniciar streamlit run).

Uso manual:
    python actualizar_diario.py

Uso programado (Windows Task Scheduler):
    Programa:   C:\\ruta\\a\\python.exe
    Argumentos: C:\\ruta\\al\\proyecto\\actualizar_diario.py
    Iniciar en: C:\\ruta\\al\\proyecto\\   (carpeta donde vive mercado_hch.db)
"""

import subprocess
import sys
import datetime
from pathlib import Path

CARPETA = Path(__file__).parent
LOG_PATH = CARPETA / "logs_actualizacion.txt"

NOTEBOOKS = [
    "actualizacion_diaria.ipynb",
    "etl_modelo_dimensional.ipynb",
]


def log(mensaje):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{ts}] {mensaje}"
    print(linea)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(linea + "\n")


def correr_notebook(nombre):
    ruta = CARPETA / nombre
    if not ruta.exists():
        log(f"⚠️  {nombre} no encontrado en {CARPETA} — se omite")
        return False

    log(f"▶️  Ejecutando {nombre}...")
    resultado = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=1800",
            "--ExecutePreprocessor.kernel_name=python3",  # fuerza este kernel, sin importar
                                                            # cuál quedó guardado en el notebook
            "--output", nombre,
            str(ruta),
        ],
        capture_output=True, text=True,
    )

    if resultado.returncode != 0:
        log(f"❌ {nombre} falló:")
        log(resultado.stderr[-3000:])
        return False

    log(f"✅ {nombre} completado")
    return True


def git_push_cambios():
    """Sube mercado_hch.db (y los notebooks con outputs nuevos) a GitHub.
    Streamlit Community Cloud redespliega solo al detectar el push."""
    archivos = ["mercado_hch.db"] + NOTEBOOKS
    archivos_existentes = [a for a in archivos if (CARPETA / a).exists()]

    def git(*args):
        return subprocess.run(["git", *args], cwd=CARPETA, capture_output=True, text=True)

    log("▶️  Subiendo cambios a GitHub...")

    r = git("add", *archivos_existentes)
    if r.returncode != 0:
        log(f"❌ git add falló: {r.stderr[-1000:]}")
        return False

    # Si no hay cambios reales (ej. el ETL corrió pero los datos no cambiaron), no hay nada que commitear
    diff = git("diff", "--cached", "--quiet")
    if diff.returncode == 0:
        log("ℹ️  Sin cambios nuevos que subir — mercado_hch.db ya estaba al día")
        return True

    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    r = git("commit", "-m", f"Actualización automática {ts}")
    if r.returncode != 0:
        log(f"❌ git commit falló: {r.stderr[-1000:]}")
        return False

    r = git("push")
    if r.returncode != 0:
        log(f"❌ git push falló: {r.stderr[-1000:]}")
        log("   (revisa credenciales de git / conexión — el commit local sí se guardó)")
        return False

    log("✅ Cambios subidos a GitHub — Streamlit Cloud debería redesplegar solo")
    return True


def main():
    log("=" * 60)
    log("Iniciando actualización diaria")

    exito_total = True
    for nb in NOTEBOOKS:
        ok = correr_notebook(nb)
        exito_total = exito_total and ok
        if not ok:
            log("Se detiene la cadena — el siguiente paso depende de este.")
            break

    if exito_total:
        log("✅ Actualización diaria completada — mercado_hch.db al día")
        exito_total = git_push_cambios()
    else:
        log("❌ Actualización incompleta — revisar el log de arriba")

    log("=" * 60)
    sys.exit(0 if exito_total else 1)


if __name__ == "__main__":
    main()
