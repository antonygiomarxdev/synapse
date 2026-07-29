# Expert Sharding Spike: Aprendizajes (2026-07-29)

## El paralelo ESP32

Slava S corrió un modelo de 28.9M parámetros en un ESP32-S3 (512KB SRAM, 8MB PSRAM, 16MB Flash) usando:

- **Per-Layer Embeddings:** La tabla de embeddings (25M de los 28.9M) almacenada en Flash lento, solo las capas activas cargadas en SRAM
- **mmap:** El runtime accede a los pesos desde Flash bajo demanda, no los carga todos en RAM
- **Cuantización 4-bit:** Reduce el modelo a ~14.9 MB total

La lección no son las técnicas específicas — es que **escribió su propio runtime porque ninguno existente soportaba su hardware.** No usó vLLM, no usó llama.cpp. Construyó exactamente lo que necesitaba.

Synapse necesita la misma aproximación para expert sharding: ningún runtime existente (vLLM, Ollama, llama.cpp) soporta "cargar solo expertos 0-19 y rechazar tokens que requieran expertos 20-39". Vamos a tener que construirlo o forkear.

Pero primero, validamos que técnicamente es posible. Eso es lo que logramos hoy.

---

## Hallazgo clave: El GGUF ya está diseñado para expert sharding

### La estructura

El formato GGUF almacena tensores de expertos con la **dimensión de expertos en el eje 0 de los datos**, transpuesta desde el orden lógico:

| Tensor | Shape lógico | Shape datos | Tipo |
|---|---|---|---|
| `blk.N.ffn_down_exps.weight` | `[512, 1536, 40]` | `[40, 1536, 420]` | Q6_K |
| `blk.N.ffn_gate_exps.weight` | `[1536, 512, 40]` | `[40, 512, 864]` | Q4_K |
| `blk.N.ffn_up_exps.weight` | `[1536, 512, 40]` | `[40, 512, 864]` | Q4_K |
| `blk.N.ffn_gate_inp.weight` | `[1536, 40]` | `[40, 1536]` | F32 |

Cada "capa" de un experto individual es completamente contigua en memoria. **Cortar `data[0:20]` nos da 20 expertos independientes sin necesidad de decodificar bloques de cuantización.**

Esto significa que **el formato GGUF ya soporta expert sharding a nivel de archivo.** Solo falta que el runtime (llama.cpp) pueda cargar un subconjunto de expertos. La data está lista.

### El splitter

`scripts/split_gguf.py` — 239 líneas de Python:

1. Lee el GGUF con `GGUFReader`
2. Para tensores de expertos: `data[start_exp:end_exp]` — numpy slicing puro
3. Para tensores compartidos (attention, embeddings, norms): copia completa
4. Escribe con `GGUFWriter`, preservando tipos de cuantización via `raw_dtype`

Resultado: Granite MoE 3B (40 expertos, 1.9 GB, 322 tensores) → 2 shards de 20 expertos, 1.06 GB c/u. Tiempo de ejecución: 6 segundos. Sin decodificar cuantización.

### Estado del splitter

| Tipo de tensor | Funciona | % del modelo |
|---|---|---|
| Q4_K expertos (`ffn_gate_exps`, `ffn_up_exps`) | ✅ Slicing directo | ~70% |
| Q6_K expertos (`ffn_down_exps`) | ✅ Slicing directo | ~28% |
| F32 compartidos (norms, output) | ✅ Copia directa | ~1% |
| Q8_0 compartido (token_embd) | ✅ Copia directa | ~0.5% |
| **F32 expertos (`ffn_gate_inp`)** | **⚠️ Shape mismatch** | **~0.5%** |

El bloqueo de F32 expertos es un mismatch de cómo `GGUFWriter.add_tensor()` maneja `raw_shape` para tensores F32 vs cómo el loader espera los datos. Afecta solo 32 tensores de 322. Es un bug conocido en la librería `gguf` (0.13.5) que no maneja correctamente la transposición de F32 cuando se pasa `raw_shape`. Solución: o parchar la librería o escribir los F32 como float16.

---

## Implicaciones para Synapse

### Arquitectura de routing

El hallazgo del GGUF confirma que la arquitectura propuesta para Synapse es correcta:

```
Coordinador (mapeo experto → worker)

Worker A: GGUF con expertos 0-19     Worker B: GGUF con expertos 20-39
         (shared layers + Q6_K/Q4_K)          (shared layers + Q6_K/Q4_K)
```

El coordinador mantiene un mapeo `{expert_id: worker_id}`. Por cada token:

1. El router del modelo en el coordinador determina qué expertos activar (top-8 de 40)
2. El coordinador envía el request a los workers que tienen esos expertos
3. Cada worker procesa con sus expertos locales
4. El coordinador combina los outputs

**Lo que falta para hacer esto real:**

1. **Runtime que cargue GGUF parcial:** Modificar llama.cpp para aceptar `--expert-ids 0-19` y cargar solo esos tensores
2. **Router remoto:** El gate/routing del MoE (la capa `ffn_gate_inp`) debe ejecutarse en el coordinador, no en el worker
3. **Combinación de outputs:** El coordinador necesita recibir outputs parciales de cada worker y combinarlos

### Dónde estamos

```
▶ Pipeline Rust↔Python↔protobuf+GPU: ✅ Validado
▶ MoE real (Granite 3B, 40 expertos): ✅ Validado
▶ GGUF expert sharding es posible: ✅ Validado (98% del modelo)
▶ Runtime con expert sharding: ❌ No existe, hay que construirlo
▶ Router remoto + combinación: ❌ No implementado
```

El paso más importante está validado: **partir un modelo MoE por expertos es técnicamente posible con las herramientas existentes.** Lo que no existe (llama.cpp modificado) lo construimos.

---

## Referencias

- ESP32-AI (Slava S): https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-developer-runs-28-9-million-parameter-model-on-usd10-esp32-s3-microcontroller
- `scripts/split_gguf.py` — splitter funcional
- GGUF format spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- llama.cpp: https://github.com/ggerganov/llama.cpp
- Modelo usado: IBM Granite 3.1 MoE 3B (granitemoe, 40 expertos, Apache 2.0)
