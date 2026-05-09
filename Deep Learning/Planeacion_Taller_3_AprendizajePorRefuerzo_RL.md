# Planeación - Taller 3: Aprendizaje por Refuerzo en Triqui

**Curso:** Técnicas de Inteligencia Artificial  
**Tema:** Aprendizaje por Refuerzo  
**Archivo base del enunciado:** `Taller_AprendizajePorRefuerzo-RL (1).pdf`  
**Entregable principal esperado:** notebook de Jupyter con código reproducible, resultados visibles, gráficas, métricas y discusión.

---

## 1. Lectura del enunciado y propósito del taller

El taller pide diseñar un agente capaz de jugar triqui con alto desempeño, partiendo de un agente base con Q-Learning y superándolo con una mejora significativa o con un algoritmo alternativo.

La pregunta que debe sostener todo el desarrollo es:

> ¿Qué enfoque de aprendizaje permite aprender mejor a jugar triqui y por qué?

La respuesta no se puede basar solo en que el agente gane algunas partidas. Debe quedar justificada con análisis teórico y evidencia experimental.

---

## 2. Metodología que vamos a conservar de los talleres anteriores

Los talleres anteriores del curso siguen una estructura bastante clara:

1. Portada académica y objetivo del taller.
2. Configuración inicial: librerías, semillas y parámetros globales.
3. Construcción de una base común para que todos los modelos se comparen de forma justa.
4. Modelo o agente base, con explicación de su funcionamiento.
5. Búsqueda o prueba de variantes.
6. Selección de los mejores candidatos usando métricas de validación.
7. Evaluación final reservada para los modelos seleccionados.
8. Tablas comparativas, gráficas y discusión técnica.
9. Resumen de entregables cubiertos.

Para este taller se seguirá el mismo flujo, pero adaptado a aprendizaje por refuerzo: entorno, agentes, entrenamiento por episodios, evaluación sin aprendizaje y comparación entre políticas.

---

## 3. Requisitos obligatorios del PDF y cómo se van a cubrir

| Requisito del enunciado | Cómo se cubrirá en el notebook |
|---|---|
| Analizar el agente base basado en Q-Learning | Se reproducirá el agente base de la sesión de RL y se evaluará antes de modificarlo. |
| Explicar por qué pierde | Se medirá contra oponentes automáticos y se revisarán partidas perdidas para detectar patrones. |
| Explicar por qué no converge | Se graficará rendimiento vs episodios y se discutirá la falta de separación entre jugador/agente, exploración fija, actualización escasa y recompensas tardías. |
| Explicar por qué aprende lentamente | Se analizará el tamaño del espacio estado-acción, recompensas terminales, exploración y ausencia de planificación. |
| Rediseñar el agente | Se probarán varios agentes candidatos en la segunda etapa. |
| Justificar cada modificación | Cada agente tendrá una celda markdown con la razón teórica de sus cambios. |
| Demostrar que supera al agente base | Los mejores candidatos se compararán contra el agente base en evaluación final. |
| Evaluar consistencia en diferentes escenarios | Se probará cuando el agente inicia, cuando inicia el oponente y con tableros iniciales parcialmente llenos. |
| Medir tasa de victoria | Se calculará win rate por agente, oponente y escenario. |
| Medir distribución de resultados | Se reportarán porcentajes de victorias, empates y derrotas. |
| Medir velocidad de aprendizaje | Se graficará rendimiento por bloques de episodios y episodios hasta estabilización. |
| Medir calidad de política | Se evaluará capacidad de evitar derrotas, bloquear amenazas y aprovechar errores. |
| Medir estabilidad | Se repetirán entrenamientos con varias semillas y se reportará media/desviación estándar. |
| Medir eficiencia | Se medirá tiempo de entrenamiento y uso aproximado de memoria. |
| Entrenar con número fijo de episodios | Se usará un protocolo común, por ejemplo 50,000 episodios por agente principal. |
| Evaluar en al menos 10,000 partidas | La evaluación final usará mínimo 10,000 partidas por combinación relevante. |
| Evaluación sin aprendizaje | Durante la fase de prueba se congelará la política y se usará acción greedy. |
| Usar semillas controladas | Se fijarán semillas de `random`, `numpy` y `torch` si se usa DQN. |
| Notebook organizado y comentado | El código irá en celdas modulares, con explicaciones en markdown natural. |
| Resultados visibles, gráficas y métricas | Se incluirán tablas, curvas de aprendizaje, barras comparativas y resumen final. |
| Video máximo 15 minutos | Se dejará una sección final con guion breve sugerido para la explicación. |

---

## 4. Librerías previstas

Se priorizarán librerías ya usadas en los ejemplos del curso:

- `numpy`: representación del tablero, estados, métricas y tablas Q.
- `random`: decisiones aleatorias controladas por semilla.
- `pandas`: consolidación de resultados en tablas.
- `matplotlib.pyplot`: curvas de aprendizaje y gráficas comparativas.
- `seaborn`: visualizaciones limpias de resultados, si está disponible.
- `time`: medición de tiempo de entrenamiento y evaluación.
- `tracemalloc` o `sys.getsizeof`: estimación simple de memoria.
- `dataclasses`: clases limpias para configuración y resultados.
- `torch`, `torch.nn`, `torch.optim`: solo si se incluye DQN como agente adicional.

No se necesita una librería externa pesada para el entorno de triqui. El tablero es pequeño y conviene implementarlo de forma transparente para poder explicar cada decisión. Si se usa un estilo tipo `gymnasium`, será solo como patrón de diseño: `reset()`, `step()`, `available_actions()` y `render()`.

---

## 5. Estructura propuesta del notebook

### 0. Portada

Incluir:

- Nombre del taller.
- Curso, universidad, profesor y estudiantes.
- Fecha.
- Pregunta guía del taller.

### 1. Objetivo y protocolo experimental

Explicar el objetivo en palabras propias:

- Diagnosticar el agente Q-Learning base.
- Rediseñar o reemplazar el enfoque.
- Comparar experimentalmente con métricas reproducibles.
- Seleccionar los mejores agentes de la segunda etapa para competir contra el agente base.

También se fijará el protocolo:

- Episodios de entrenamiento: `50_000`.
- Evaluación final: mínimo `10_000` partidas por escenario.
- Semillas: por ejemplo `[7, 21, 42, 84, 123]`.
- Evaluación sin aprendizaje: `epsilon = 0` o modo greedy.

### 2. Librerías, semillas y configuración global

Crear funciones:

- `set_seed(seed)`.
- Configuración global de episodios, partidas de evaluación, ventanas de suavizado y lista de escenarios.

La intención es que cambiar el número de episodios o semillas no obligue a reescribir el notebook.

### 3. Entorno de triqui

Implementar una clase o bloque modular con:

- Estado: tablero `3 x 3`, con valores `1`, `-1`, `0`.
- Acciones: posiciones vacías del tablero.
- Validación de jugadas.
- Detección de ganador por filas, columnas y diagonales.
- Detección de empate.
- Conversión de tablero a estado hashable: tupla plana de 9 elementos.
- Opción de iniciar desde tablero vacío o desde estados parcialmente llenos válidos.

Funciones sugeridas:

- `initial_state()`.
- `available_actions(board)`.
- `check_winner(board)`.
- `is_terminal(board)`.
- `state_to_tuple(board)`.
- `make_move(board, action, player)`.
- `generate_partial_board(rng, min_moves=1, max_moves=4)`.
- `render_board(board)`.

### 4. Oponentes automáticos

Definir oponentes para evaluación y entrenamiento:

1. **Oponente aleatorio:** elige cualquier casilla disponible.
2. **Agente base:** política aprendida por el Q-Learning base.
3. **Oponente heurístico simple:** gana si puede, bloquea si debe, si no juega centro/esquina/azar.

El PDF exige evaluar contra oponente aleatorio y agente base. El heurístico se usará como prueba adicional de calidad de política, porque ayuda a ver si el agente solo gana contra errores obvios o si también evita derrotas.

### 5. Reproducción del agente base Q-Learning

Partir del código de ejemplo de la sesión de aprendizaje por refuerzo y dejarlo organizado en funciones.

Aspectos importantes que se deben conservar para que sea realmente el agente base:

- Q-table con diccionario.
- `alpha = 0.1`.
- `gamma = 0.9`.
- `epsilon = 0.2`.
- Recompensa terminal simple: victoria `+1`, derrota `-1`, empate `0`.

Luego se entrena y evalúa con el mismo protocolo que los demás agentes.

### 6. Diagnóstico del agente base

Esta parte debe responder de forma directa:

1. **Por qué pierde**
   - Puede no bloquear amenazas inmediatas.
   - Puede elegir siempre la primera acción con Q máximo cuando hay empates.
   - Puede aprender valores contradictorios si no se separa bien la perspectiva de X y O.

2. **Por qué no converge**
   - La exploración permanece fija.
   - El aprendizaje solo recibe señal fuerte al final de la partida.
   - El mismo esquema de decisión puede usarse para turnos distintos sin normalizar la perspectiva del jugador.
   - No hay evaluación periódica que permita observar estabilización.

3. **Por qué aprende lentamente**
   - Hay muchas combinaciones de tablero aunque el juego sea pequeño.
   - Muchas partidas terminan con recompensas tardías.
   - El agente no reutiliza simetrías del tablero.
   - No planifica con experiencias simuladas.

Evidencia prevista:

- Curva de win rate durante entrenamiento.
- Distribución de victorias/empates/derrotas.
- Ejemplos de partidas perdidas o decisiones débiles.
- Tabla de tiempo y tamaño de Q-table.

### 7. Segunda etapa: agentes candidatos a probar

Se probarán varios agentes y luego se escogerán los de mejor resultado para competir formalmente con el agente base.

#### Agente A - Q-Learning mejorado

Cambios:

- Separar correctamente la perspectiva del agente cuando juega como X o como O.
- Usar `epsilon` decreciente.
- Romper empates entre acciones máximas de forma aleatoria.
- Ajustar recompensas:
  - victoria: `+1`
  - derrota: `-1`
  - empate: `+0.2`
  - bloquear una victoria inmediata del rival: recompensa pequeña positiva
  - permitir una victoria inmediata del rival: penalización pequeña

Justificación:

Q-Learning puede funcionar bien en triqui porque el espacio es discreto y pequeño, pero necesita una señal de recompensa menos pobre y una exploración que disminuya cuando la política mejora.

#### Agente B - SARSA

Cambios:

- Actualización on-policy:
  `Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))`
- Misma representación de estados y recompensas que el Q-Learning mejorado.

Justificación:

SARSA aprende considerando la acción que realmente tomará bajo su política exploratoria. En juegos con riesgo de derrota inmediata, esto puede generar políticas más prudentes que Q-Learning puro.

#### Agente C - Dyna-Q

Cambios:

- Mantener Q-Learning como aprendizaje directo.
- Guardar un modelo de transiciones observadas.
- Hacer pasos de planificación por cada jugada real.

Justificación:

Dyna-Q puede aprender más rápido porque no depende solo de partidas nuevas: reutiliza experiencias pasadas como simulaciones. Esto apunta directamente al requisito de velocidad de aprendizaje.

#### Agente D - DQN tabularizado o red pequeña con PyTorch

Este agente será opcional si el tiempo alcanza o si los agentes tabulares no muestran diferencias claras.

Cambios:

- Representar el tablero como vector de 9 entradas.
- Red pequeña: `9 -> 64 -> 64 -> 9`.
- Salida: valor Q para cada casilla.
- Máscara de acciones inválidas.
- Replay buffer y target network, siguiendo el estilo del ejemplo de DQN visto en clase.

Justificación:

DQN permite probar un enfoque de aproximación funcional. En triqui puede ser innecesario por el tamaño pequeño del problema, pero sirve como comparación con técnicas de Deep RL.

### 8. Selección de los mejores agentes de la segunda etapa

Antes de competir contra el agente base en la evaluación final, los candidatos se compararán en una fase de selección.

Métricas de selección:

- Win rate contra oponente aleatorio.
- Porcentaje de derrotas.
- Win rate cuando el agente juega primero y cuando juega segundo.
- Rendimiento en tableros parcialmente llenos.
- Episodios necesarios para estabilizarse.
- Tiempo de entrenamiento.
- Variación entre semillas.

Criterio principal:

1. Priorizar menor tasa de derrotas.
2. Luego mayor tasa de victorias.
3. Luego mayor estabilidad entre semillas.
4. Luego menor tiempo de entrenamiento.

Se seleccionarán los mejores 2 o 3 agentes para la comparación final contra el agente base.

### 9. Comparación experimental final

Agentes a comparar:

- Agente base Q-Learning.
- Mejores agentes seleccionados de la segunda etapa.

Oponentes:

- Oponente aleatorio.
- Agente base.
- Oponente heurístico como evaluación adicional.

Escenarios:

1. Agente inicia la partida.
2. Oponente inicia la partida.
3. Tablero parcialmente lleno y turno del agente.
4. Tablero parcialmente lleno y turno del oponente.

Cada combinación debe ejecutarse sin aprendizaje y con al menos `10_000` partidas cuando sea parte de la evaluación principal.

### 10. Métricas y gráficas

Tablas:

- Resultados por agente, oponente y escenario.
- Media y desviación estándar por semillas.
- Tiempo de entrenamiento y tamaño de política/Q-table.
- Ranking de agentes.

Gráficas:

- Curvas de aprendizaje: win rate vs episodios.
- Barras apiladas: victorias, empates y derrotas.
- Barras con error: win rate medio y desviación estándar.
- Comparación de tiempo de entrenamiento.
- Si aplica, evolución del tamaño de Q-table.

Métricas calculadas:

- `win_rate = victorias / total_partidas`.
- `draw_rate = empates / total_partidas`.
- `loss_rate = derrotas / total_partidas`.
- episodios hasta estabilización.
- tiempo de entrenamiento.
- memoria aproximada.

### 11. Discusión técnica esperada

La discusión debe responder la pregunta guía del taller con evidencia.

Puntos a cubrir:

- Si el mejor agente gana más que el base y en qué escenarios.
- Si evita derrotas mejor que el base.
- Si aprende más rápido.
- Si el resultado es estable o depende demasiado de la semilla.
- Qué agente ofrece mejor balance entre desempeño y simplicidad.
- Por qué DQN puede no ser necesario si los métodos tabulares ya resuelven bien el juego.
- Qué limitaciones quedan: tablero pequeño, oponentes automáticos, ausencia de humanos reales durante evaluación masiva.

### 12. Conclusiones

La conclusión no debe sonar genérica. Debe salir de las tablas:

- Mejor agente encontrado.
- Evidencia principal que lo respalda.
- Qué cambio fue más importante.
- Qué enfoque no valió tanto la pena y por qué.

Ejemplo de forma esperada:

> En este problema, el mejor balance lo obtuvo Dyna-Q/Q-Learning mejorado porque redujo las derrotas casi a cero, mantuvo una tasa alta de victorias contra el oponente aleatorio y fue más estable entre semillas que el agente base. La mejora no vino solo de cambiar hiperparámetros, sino de corregir la perspectiva del jugador, controlar la exploración y enriquecer la señal de recompensa.

La frase exacta se escribirá después de ejecutar los experimentos.

### 13. Resumen de entregables cubiertos

Cerrar el notebook con una tabla de verificación. Después de la implementación, el estado queda así:

| Entregable | Estado |
|---|---|
| Notebook organizado en celdas | Cubierto en `Taller_3_AprendizajePorRefuerzo_Triqui.ipynb` |
| Código reproducible con semillas | Cubierto con `ExperimentConfig` y semillas `(7, 21, 42)` |
| Diagnóstico del agente base | Cubierto con métricas, curvas y análisis de fallas |
| Nuevo enfoque justificado | Cubierto con Q-Learning mejorado, SARSA y Dyna-Q |
| Comparación experimental | Cubierto contra oponente aleatorio y agente base |
| Gráficas y métricas visibles | Cubierto con curvas de aprendizaje, distribución de resultados y barras comparativas |
| Discusión de resultados | Cubierto con interpretación y conclusión final |
| Guion para video de máximo 15 minutos | Cubierto en la sección final del notebook |

### 14. Extensión táctica: cerrar jugadas del oponente

Sí es posible cambiar el énfasis del agente para que no solo "evite perder", sino que cierre activamente las jugadas del oponente. En triqui esto significa detectar amenazas y responder con una acción que le quite al rival una línea prometedora.

La versión actual ya mide una parte de esa idea en la sección de calidad de política:

1. Si el agente tiene victoria inmediata, debe jugarla.
2. Si el rival tiene victoria inmediata, el agente debe bloquearla.

Para convertir esto en una sección más fuerte del taller, se puede agregar una métrica específica de **cierre táctico**:

| Métrica | Qué mide |
|---|---|
| `block_rate` | Porcentaje de veces que bloquea una victoria inmediata del rival. |
| `fork_block_rate` | Porcentaje de veces que evita que el rival cree una doble amenaza. |
| `threat_reduction_rate` | Cuánto reduce el número de líneas abiertas del oponente después de jugar. |
| `counter_threat_rate` | Porcentaje de jugadas donde, además de cerrar al rival, crea una amenaza propia. |

La implementación se puede hacer con funciones auxiliares:

```python
def open_lines(board, player):
    # Cuenta líneas donde el jugador todavía puede completar tres fichas.
    ...

def creates_fork(board, action, player):
    # Evalúa si una jugada crea dos o más amenazas de victoria inmediata.
    ...

def closing_score(before, after, opponent):
    # Mide cuántas líneas abiertas o amenazas pierde el oponente.
    ...
```

Luego se puede modificar la recompensa táctica:

```python
reward += 0.05  # bloquear victoria inmediata
reward += 0.03  # reducir líneas abiertas del rival
reward += 0.06  # evitar fork del rival
reward += 0.04  # cerrar al rival y crear amenaza propia
```

La idea importante es no premiar solo el empate. Un agente puede empatar jugando pasivamente, pero un agente con cierre táctico debe mostrar que interrumpe planes concretos del oponente: bloquea líneas, evita dobles amenazas y transforma una defensa en una amenaza propia.

Para el notebook, esta extensión quedaría después de la sección **Calidad de la política**, con una tabla comparando:

- Q-Learning base.
- SARSA.
- Dyna-Q.
- Variante táctica con recompensa de cierre.

El criterio de éxito sería que la variante táctica aumente `block_rate`, `fork_block_rate` y `threat_reduction_rate`, sin perder demasiado `win_rate`.

---

## 6. Diseño limpio del código

Para que el notebook no se vuelva una colección de bloques sueltos, el código se organizará con funciones y clases pequeñas:

```python
@dataclass
class TrainingConfig:
    episodes: int
    alpha: float
    gamma: float
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    seed: int
```

Clases sugeridas:

- `TicTacToeEnv`
- `RandomOpponent`
- `HeuristicOpponent`
- `BaseQLearningAgent`
- `ImprovedQLearningAgent`
- `SarsaAgent`
- `DynaQAgent`
- `DQNAgent` si se incluye

Funciones comunes:

- `train_agent(agent, env, opponent, config)`.
- `evaluate_agent(agent, env, opponent, games, seed, scenario)`.
- `run_experiment(agent_factory, seeds, config)`.
- `summarize_results(results_df)`.
- `plot_learning_curves(history_df)`.
- `plot_outcome_distribution(results_df)`.

El objetivo es que todos los agentes pasen por el mismo pipeline y que la comparación sea justa.

---

## 7. Estilo de explicación en el notebook

Las celdas markdown deben sonar como un informe técnico escrito por estudiantes, no como texto automático.

Buen tono:

- Explicar qué se está probando y por qué importa.
- Conectar cada decisión con el comportamiento del juego.
- No llenar el notebook con comentarios obvios dentro del código.
- Usar comentarios solo donde ayuden a entender una decisión.

Evitar:

- "En esta celda importamos librerías" repetido sin aportar nada.
- Comentarios tipo "se crea variable x".
- Conclusiones antes de tener resultados.
- Decir que un agente es mejor sin mostrar la métrica que lo respalda.

---

## 8. Orden de trabajo recomendado

1. Crear el notebook con portada y protocolo.
2. Implementar entorno de triqui y pruebas rápidas del entorno.
3. Reproducir agente base.
4. Crear evaluación común sin aprendizaje.
5. Diagnosticar el agente base con métricas iniciales.
6. Implementar Q-Learning mejorado.
7. Implementar SARSA.
8. Implementar Dyna-Q.
9. Incluir DQN solo si el tiempo y el entorno de ejecución lo permiten.
10. Ejecutar selección de candidatos con varias semillas.
11. Comparar mejores agentes contra el agente base.
12. Generar gráficas y tablas.
13. Escribir discusión y conclusiones.
14. Agregar tabla final de entregables.
15. Preparar guion corto para el video.

---

## 9. Riesgos y decisiones anticipadas

| Riesgo | Decisión |
|---|---|
| Entrenar DQN puede tomar más tiempo que los métodos tabulares | Dejar DQN como extensión opcional, no como requisito central. |
| El agente base puede verse artificialmente malo por errores del ejemplo | Se mantendrá como base, pero el diagnóstico explicará justamente sus limitaciones. |
| El triqui óptimo tiende al empate | No se medirá solo victoria; se dará mucho peso a evitar derrotas y estabilidad. |
| Los resultados pueden variar por semilla | Se usarán varias semillas y se reportará desviación estándar. |
| Evaluar demasiadas combinaciones puede ser lento | Separar fase de selección y fase final. Solo los mejores pasan a evaluación completa. |

---

## 10. Criterio de finalización del taller

El taller estará completo cuando el notebook pueda ejecutarse de inicio a fin y deje visibles:

- Diagnóstico del agente base.
- Al menos tres agentes candidatos probados en la segunda etapa.
- Selección de los mejores candidatos con métricas.
- Competencia final contra el agente base.
- Evaluación contra oponente aleatorio y agente base.
- Escenarios con agente iniciando, oponente iniciando y estados parcialmente llenos.
- Al menos 10,000 partidas por evaluación principal.
- Semillas controladas.
- Curvas, tablas y discusión.
- Conclusión que responda la pregunta guía con evidencia.
