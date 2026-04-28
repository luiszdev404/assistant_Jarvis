# Jarvis2

Jarvis2 es un asistente de inteligencia artificial modular y orientado a eventos, diseñado específicamente para entornos Linux Fedora con escritorio GNOME. Desarrollado con una arquitectura limpia y un registro de habilidades extensible, utiliza la API de Gemini Live para proporcionar capacidades conversacionales avanzadas e integración con el sistema directamente desde la terminal.

## Arquitectura Principal

El sistema está construido sobre una arquitectura asíncrona moderna que prioriza la modularidad y la capacidad de respuesta:

*   **Núcleo Asíncrono Orientado a Eventos**: Orquesta las operaciones de manera concurrente, asegurando que la interfaz de la terminal se mantenga receptiva durante tareas prolongadas, como búsquedas web o solicitudes a la API.
*   **Registro de Habilidades Unificado**: Elimina las estructuras de código monolíticas al desacoplar las capacidades funcionales en módulos independientes. El registro carga y administra dinámicamente el acceso a las integraciones del sistema.
*   **Integración con Gemini Live API**: Actúa como el nodo central de inteligencia, procesando el lenguaje natural y enrutando las solicitudes complejas a los modelos de Gemini para generar respuestas rápidas y precisas.
*   **Interfaz de Terminal (Curses)**: Proporciona una interfaz ligera, fluida y libre de distracciones, compatible con todos los emuladores de terminal estándar y terminales integradas en entornos de desarrollo (IDE).

## Habilidades Disponibles

El diseño modular incluye actualmente las siguientes capacidades:

*   **Control de Navegador**: Navegación programática e interacción con navegadores web.
*   **Gestión de Aplicaciones**: Ejecución y control de aplicaciones locales dentro del entorno GNOME (`open_app`).
*   **Investigador Técnico**: Análisis técnico avanzado y síntesis de datos para consultas de programación y desarrollo.
*   **Búsqueda Web**: Recuperación de información en tiempo real y resumen de contenidos de internet.
*   **Integración con YouTube**: Búsqueda de videos y control de reproducción.

## Requisitos Previos

*   Sistema Operativo: Linux Fedora con entorno de escritorio GNOME.
*   Python: Versión 3.10 o superior.
*   Clave válida para la API de Google Gemini.

## Instalación

1.  **Clonar el repositorio**
    ```bash
    git clone <url_del_repositorio>
    cd Jarvis2
    ```

2.  **Inicializar el entorno virtual**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```

## Configuración

La seguridad y la configuración del entorno se gestionan mediante archivos JSON locales para evitar la exposición accidental de credenciales.

1.  Copiar la plantilla de configuración de ejemplo:
    ```bash
    cp config/api_keys.example.json config/api_keys.json
    ```

2.  Editar el archivo `config/api_keys.json` e insertar su clave de API de Gemini:
    ```json
    {
      "gemini_api_key": "SU_CLAVE_API_DE_GEMINI_AQUI"
    }
    ```
    *Nota: El archivo `api_keys.json` está explícitamente ignorado por el control de versiones (.gitignore) para garantizar la seguridad.*

## Uso

Para inicializar el asistente, asegúrese de que su entorno virtual esté activo y ejecute el punto de entrada principal:

```bash
python main.py
```

Interactúe con el asistente a través del campo de entrada estático en la parte inferior de la interfaz de la terminal. Utilice las señales de interrupción estándar (Ctrl+C) para finalizar la sesión de manera segura.

## Desarrollo y Extensibilidad

La arquitectura del Registro de Habilidades permite la implementación sencilla de nuevas capacidades. Para agregar una nueva habilidad:

1.  Cree un nuevo módulo de Python en el directorio `skills/` que herede de la clase base de habilidades.
2.  Implemente los métodos de ejecución asíncrona requeridos.
3.  Registre el nuevo módulo de habilidad en la configuración principal (`core/skill_registry.py`) para ponerlo a disposición del motor de ejecución.
