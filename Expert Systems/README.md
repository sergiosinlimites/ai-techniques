# README - Talleres de Sistemas Expertos

Guia para desarrollar el notebook final de los talleres de Sistemas Expertos del curso TIA.

## Material revisado

- `TIA_20260212_Sistemas_Expertos_Parte-UNO.pdf`, paginas 56-68.
- `TIA_20260212_Sistemas_Expertos_Parte-UNO.ipynb`, ejemplos de sistemas expertos, encadenamiento, reconocimiento de voz y sintesis de voz.
- `TIA_20260217_Sistemas_Expertos_Parte-DOS.pdf`, paginas 82-85.
- `TIA_20260217_Sistemas_Expertos_Parte-DOS.ipynb`, ejemplos de factores de certeza, MYCIN, inferencia bayesiana, logica difusa y modulo de explicacion conceptual.

## Estrategia recomendada

La mejor forma de cumplir ambos talleres con una sola entrega es desarrollar un notebook llamado, por ejemplo:

`Taller_Sistema_Experto_Piezas_Mecanicas_Voz_FC.ipynb`

Tema propuesto:

**Chatbot experto por voz para verificar piezas mecanicas usando reglas, factores de certeza MYCIN y modulo de explicacion.**

Esta integracion funciona porque:

- El Taller 1 exige un chatbot experto con interfaz por voz, reglas en logica de primer orden, motor de inferencia, diagnostico o recomendacion y sintesis de voz.
- El Taller 2 exige verificar piezas mecanicas, manejar incertidumbre con factores de certeza, acumular evidencias parciales y explicar razones de aceptacion o rechazo.
- El profesor indico que, si se integra manejo de incertidumbre y modulo de explicacion en la tarea 1, no es necesario hacer la tarea 2 por separado.

Con esta solucion integrada, el dominio libre del Taller 1 sera el dominio del Taller 2: verificacion de piezas mecanizadas.

## Objetivos del Taller 1

El sistema debe ser un chatbot interactivo que:

1. Capture respuestas del usuario mediante reconocimiento de voz.
2. Procese la informacion con un sistema experto basado en reglas en logica de primer orden.
3. Genere un diagnostico o recomendacion razonada.
4. Presente el resultado mediante sintesis de voz.

Criterios minimos:

- 8 reglas como minimo.
- Al menos 2 reglas encadenadas.
- 8 variables de entrada.
- 1 diagnostico final no trivial.
- Codigo funcional y comentado en notebook.
- Ejemplos de ejecucion.
- Explicacion del motor de inferencia.
- Video de maximo 10 minutos explicando el codigo y mostrando ejecuciones.

## Objetivos del Taller 2

El sistema debe verificar piezas mecanizadas de tipo:

- Eje.
- Brida.
- Tornillo.

Para cada pieza se deben evaluar:

- Diametro.
- Longitud.
- Rugosidad.
- Material.
- Grietas visibles.

Tolerancias suministradas:

| Tipo | Diametro (mm) | Longitud (mm) | Rugosidad maxima (Ra) | Material |
| --- | --- | --- | --- | --- |
| Eje | 49.8 - 50.2 | 99.7 - 100.3 | 2.0 | Acero |
| Brida | 80.0 - 80.5 | 19.5 - 20.5 | 2.0 | Aluminio |
| Tornillo | 9.8 - 10.2 | 19.0 - 21.0 | 1.5 | Acero |

Factores de certeza por evidencia:

| Evidencia de fallo | FC |
| --- | ---: |
| Diametro fuera de tolerancia | 0.4 |
| Longitud fuera de tolerancia | 0.3 |
| Rugosidad excesiva | 0.3 |
| Material incorrecto | 0.6 |
| Grietas visibles | 0.7 |

Formula de combinacion MYCIN:

```text
FC_total = FC_previo + (1 - FC_previo) * FC_nuevo
```

El sistema debe mostrar para cada pieza:

- Factor de certeza acumulado de rechazo.
- Razones especificas de posible fallo.
- Mensaje final: aceptar, rechazar o revisar.

## Notebook propuesto

El notebook final debe tener una estructura clara, parecida a los notebooks de clase:

1. Portada y objetivo.
2. Descripcion del problema.
3. Representacion formal del conocimiento.
4. Variables de entrada.
5. Base de hechos.
6. Base de reglas.
7. Motor de inferencia hacia adelante.
8. Manejo de incertidumbre con factores de certeza.
9. Modulo de explicacion.
10. Procesamiento de lenguaje natural basico.
11. Reconocimiento de voz.
12. Sintesis de voz.
13. Pruebas y validacion.
14. Conclusiones y preguntas de analisis.

## Variables de entrada recomendadas

Para cumplir las 8 variables del Taller 1 y mantener el dominio del Taller 2:

1. `tipo_pieza`: eje, brida o tornillo.
2. `diametro`: valor numerico en mm.
3. `longitud`: valor numerico en mm.
4. `rugosidad`: valor numerico Ra.
5. `material`: acero, aluminio u otro.
6. `grietas_visibles`: si/no.
7. `deformacion_visible`: si/no.
8. `uso_critico`: si/no.

Las primeras cinco variables vienen directamente del Taller 2. Las tres ultimas amplian el dominio para obtener un diagnostico mas completo y no trivial.

## Representacion formal

Usar hechos tipo predicado:

```text
pieza(eje)
diametro(eje_1, 50.4)
longitud(eje_1, 100.0)
rugosidad(eje_1, 1.8)
material(eje_1, acero)
grietas_visibles(eje_1, no)
deformacion_visible(eje_1, si)
uso_critico(eje_1, si)
```

Ejemplos de predicados derivados:

```text
diametro_fuera_tolerancia(x)
longitud_fuera_tolerancia(x)
rugosidad_excesiva(x)
material_incorrecto(x)
grietas_detectadas(x)
deformacion_detectada(x)
dimension_fuera_especificacion(x)
integridad_comprometida(x)
pieza_rechazada(x)
pieza_aceptada(x)
requiere_revision(x)
```

## Reglas minimas recomendadas

Definir al menos 8 reglas. Se recomienda usar mas de 8 para que el sistema sea robusto.

| ID | Regla | FC sugerido |
| --- | --- | ---: |
| R1 | Si el diametro esta fuera de tolerancia, entonces hay fallo de diametro | 0.4 |
| R2 | Si la longitud esta fuera de tolerancia, entonces hay fallo de longitud | 0.3 |
| R3 | Si la rugosidad supera el maximo permitido, entonces hay fallo de rugosidad | 0.3 |
| R4 | Si el material no coincide con el especificado, entonces hay fallo de material | 0.6 |
| R5 | Si hay grietas visibles, entonces hay fallo por grietas | 0.7 |
| R6 | Si hay deformacion visible, entonces hay fallo geometrico | 0.5 |
| R7 | Si hay fallo de diametro y fallo de longitud, entonces hay dimension fuera de especificacion | 0.8 |
| R8 | Si hay fallo de rugosidad o grietas visibles, entonces hay integridad superficial comprometida | 0.8 |
| R9 | Si hay fallo de material y uso critico, entonces hay riesgo alto de operacion | 0.9 |
| R10 | Si hay dimension fuera de especificacion o integridad comprometida o riesgo alto, entonces la pieza debe rechazarse | 0.9 |
| R11 | Si solo hay evidencia leve acumulada, entonces la pieza requiere revision manual | 0.5 |
| R12 | Si no hay fallos detectados, entonces la pieza puede aceptarse | 1.0 |

Reglas encadenadas:

- Encadenamiento 1: `diametro_fuera_tolerancia` + `longitud_fuera_tolerancia` -> `dimension_fuera_especificacion` -> `pieza_rechazada`.
- Encadenamiento 2: `rugosidad_excesiva` o `grietas_detectadas` -> `integridad_comprometida` -> `pieza_rechazada`.
- Encadenamiento 3 opcional: `material_incorrecto` + `uso_critico` -> `riesgo_alto_operacion` -> `pieza_rechazada`.

## Manejo de incertidumbre

Cada evidencia debe producir un factor de certeza. Para reglas encadenadas:

```text
FC_conclusion = min(FC_antecedentes) * FC_regla
```

Si varias evidencias apoyan la misma conclusion, combinar con MYCIN:

```python
def combinar_fc(fc_previo, fc_nuevo):
    return fc_previo + (1 - fc_previo) * fc_nuevo
```

Recomendacion importante:

- Registrar que reglas ya fueron aplicadas para no sumar la misma evidencia infinitamente.
- Guardar una traza de inferencia con regla, premisas, conclusion y FC.
- Mantener los FC en el rango `[0, 1]` para este taller, porque las evidencias del enunciado son de rechazo.

## Criterio de decision

Como el enunciado no define un umbral exacto, proponer y documentar uno:

| FC acumulado de rechazo | Decision |
| ---: | --- |
| `FC < 0.40` | Aceptar pieza |
| `0.40 <= FC < 0.60` | Requiere revision manual |
| `FC >= 0.60` | Rechazar pieza |

Este criterio debe estar explicado en el notebook y usado de forma consistente en las pruebas.

## Modulo de explicacion

El sistema debe responder no solo que decision tomo, sino por que.

Debe guardar una traza como:

```python
{
    "regla": "R5",
    "premisas": ["grietas_visibles"],
    "conclusion": "fallo_grietas",
    "fc": 0.7,
    "explicacion": "Se detectaron grietas visibles, evidencia fuerte de rechazo."
}
```

La salida final debe incluir:

- Decision final.
- FC acumulado.
- Lista de razones.
- Reglas activadas.
- Cadena de inferencia.

Ejemplo de explicacion esperada:

```text
Decision: RECHAZAR pieza.
FC acumulado de rechazo: 0.82
Razones:
- Diametro fuera de tolerancia: FC 0.40
- Grietas visibles: FC 0.70
- Integridad comprometida inferida por R8
Explicacion:
R1 detecto fallo de diametro.
R5 detecto grietas.
R8 infirio integridad comprometida.
R10 infirio rechazo final.
```

## Procesamiento de lenguaje natural basico

Implementar un extractor simple basado en normalizacion, palabras clave y expresiones regulares.

Entrada de texto ejemplo:

```text
La pieza es un eje con diametro 50.4, longitud 100.1, rugosidad 1.9,
material acero, con grietas visibles, sin deformacion y uso critico.
```

El extractor debe convertirlo a:

```python
{
    "tipo_pieza": "eje",
    "diametro": 50.4,
    "longitud": 100.1,
    "rugosidad": 1.9,
    "material": "acero",
    "grietas_visibles": True,
    "deformacion_visible": False,
    "uso_critico": True
}
```

Tambien se puede permitir entrada por formulario o diccionario para validar el motor sin depender del microfono.

## Voz y sintesis

Para reconocimiento de voz:

```python
import speech_recognition as sr

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    audio = recognizer.listen(source)

texto = recognizer.recognize_google(audio, language="es-ES")
```

Para sintesis de voz:

```python
import pyttsx3

engine = pyttsx3.init()
engine.say("La pieza debe rechazarse por grietas visibles.")
engine.runAndWait()
```

Como el reconocimiento de voz puede fallar por microfono, ruido o conexion, el notebook debe incluir un modo alternativo:

```python
MODO_VOZ = False

if MODO_VOZ:
    texto = reconocer_voz()
else:
    texto = input("Describa la pieza: ")
```

## Pruebas de validacion

El notebook debe incluir pruebas visibles con resultados esperados.

### Prueba 1: pieza aceptada

Entrada:

```python
{
    "tipo_pieza": "eje",
    "diametro": 50.0,
    "longitud": 100.0,
    "rugosidad": 1.5,
    "material": "acero",
    "grietas_visibles": False,
    "deformacion_visible": False,
    "uso_critico": False
}
```

Resultado esperado:

- Sin fallos detectados.
- FC rechazo cercano a 0.
- Decision: aceptar.

### Prueba 2: rechazo por material y grietas

Entrada:

```python
{
    "tipo_pieza": "brida",
    "diametro": 80.2,
    "longitud": 20.0,
    "rugosidad": 1.8,
    "material": "acero",
    "grietas_visibles": True,
    "deformacion_visible": False,
    "uso_critico": True
}
```

Resultado esperado:

- Fallo por material incorrecto.
- Fallo por grietas visibles.
- FC acumulado alto.
- Decision: rechazar.

### Prueba 3: rechazo por encadenamiento dimensional

Entrada:

```python
{
    "tipo_pieza": "tornillo",
    "diametro": 10.5,
    "longitud": 21.5,
    "rugosidad": 1.2,
    "material": "acero",
    "grietas_visibles": False,
    "deformacion_visible": False,
    "uso_critico": False
}
```

Resultado esperado:

- Fallo de diametro.
- Fallo de longitud.
- Inferencia de dimension fuera de especificacion.
- Decision: revision o rechazo, segun el FC acumulado definido.

### Prueba 4: revision manual

Entrada:

```python
{
    "tipo_pieza": "eje",
    "diametro": 50.3,
    "longitud": 100.0,
    "rugosidad": 1.7,
    "material": "acero",
    "grietas_visibles": False,
    "deformacion_visible": False,
    "uso_critico": False
}
```

Resultado esperado:

- Fallo leve por diametro.
- FC alrededor de 0.4.
- Decision: revision manual.

### Prueba 5: procesamiento de texto

Entrada:

```text
Es un eje, diametro 50.4, longitud 100.2, rugosidad 2.3,
material acero, sin grietas, sin deformacion, uso critico.
```

Resultado esperado:

- Extraer correctamente las 8 variables.
- Detectar diametro fuera de tolerancia.
- Detectar rugosidad excesiva.
- Ejecutar inferencia completa.

## Validacion tecnica

Antes de entregar:

- Ejecutar el notebook completo desde cero.
- Verificar que todas las celdas corran en orden.
- Comprobar que hay al menos 8 reglas definidas.
- Comprobar que hay al menos 2 reglas encadenadas.
- Comprobar que hay 8 variables de entrada.
- Comprobar que se calcula y muestra `FC acumulado`.
- Comprobar que la formula MYCIN se aplica cuando hay mas de una evidencia.
- Comprobar que se imprime una explicacion con reglas activadas.
- Comprobar que existen ejemplos para eje, brida y tornillo.
- Comprobar que existe modo de texto aunque falle la voz.
- Comprobar que la sintesis de voz lee el resultado final o, como minimo, que la funcion queda implementada y documentada.

## Preguntas de analisis que deben responderse

Incluir una seccion final de respuestas breves:

1. Importancia del tema seleccionado: control de calidad, seguridad industrial y reduccion de rechazos no justificados.
2. Limitaciones del reconocimiento de voz: ruido, acentos, errores de transcripcion, dependencia de microfono o conexion.
3. Formalizacion del lenguaje natural: convertir frases en hechos y predicados logicos.
4. Ventajas de la logica simbolica: trazabilidad, explicabilidad y reglas auditables.
5. Escalamiento industrial: agregar sensores, historicos, bases de datos, mas tipos de piezas, calibracion y monitoreo en linea.

## Entregables finales

Subir:

- Notebook `.ipynb` funcional, documentado y comentado.
- Si se prefiere, un `.py` auxiliar con funciones del motor experto.
- Video de maximo 10 minutos.
- Este README como guia de alcance y validacion.

## Guion sugerido para el video

Duracion maxima: 10 minutos.

1. Presentar el objetivo del sistema integrado.
2. Mostrar las tolerancias y factores de certeza usados.
3. Explicar la representacion de hechos y reglas.
4. Explicar el motor de inferencia hacia adelante.
5. Explicar como se combinan los factores de certeza con MYCIN.
6. Mostrar el modulo de explicacion.
7. Ejecutar un caso aceptado.
8. Ejecutar un caso rechazado.
9. Ejecutar un caso por voz o mostrar el modo texto si el microfono falla.
10. Cerrar con limitaciones y mejoras.

## Checklist final

- [ ] El notebook tiene portada, objetivos y descripcion del problema.
- [ ] El dominio es verificacion de piezas mecanicas.
- [ ] Se usan eje, brida y tornillo.
- [ ] Se usan las tolerancias del enunciado.
- [ ] Hay 8 variables de entrada.
- [ ] Hay 8 reglas o mas.
- [ ] Hay al menos 2 reglas encadenadas.
- [ ] Se implementa encadenamiento hacia adelante.
- [ ] Se implementa combinacion MYCIN.
- [ ] Se reporta FC acumulado de rechazo.
- [ ] Se muestran razones especificas de fallo.
- [ ] Se genera decision final.
- [ ] Hay modulo de explicacion.
- [ ] Hay reconocimiento de voz o modo alternativo documentado.
- [ ] Hay sintesis de voz o funcion documentada.
- [ ] Hay pruebas para aceptar, rechazar y revisar.
- [ ] El notebook corre completo desde cero.
- [ ] El video dura maximo 10 minutos.
