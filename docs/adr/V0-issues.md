# V0 — Issues (Semana 0-4)

> Crear estos issues en GitHub: https://github.com/antonygiomarxdev/synapse/issues

---

## V0-1: Job model + async API

**Prioridad:** Semana 1
**Label:** `v0`, `feature`

Definir la entidad `Job` y exponer endpoints asíncronos:
- `POST /v1/jobs` → `202 Accepted` con `job_id`
- `GET /v1/jobs/{id}` → estado + URL de resultado

**Scope:**
- Struct `Job`: id, prompts, model_id, priority, deadline, status, result_url, created_at
- Trait `JobStore`: save, find_by_id, list, update_status
- Implementación in-memory para V0
- Rutas del gateway + request/response types
- Test de integración: submit job, poll hasta completar

**Non-goals:** Persistencia, cancelación, paginación, auth

**Acceptance:**
- `POST /v1/jobs` con payload válido → `202` + `job_id`
- `GET /v1/jobs/{id}` → estado actual
- 100 jobs concurrentes, todos recuperables
- Payload inválido → `400` con error

---

## V0-2: Scheduler mínimo

**Prioridad:** Semana 1
**Label:** `v0`, `feature`

Despachar tareas de jobs a workers conocidos con leases, timeouts, y reintentos.

**Scope:**
- `Scheduler` struct: recibe jobs del gateway, despacha a workers, recoge resultados
- `Lease`: cada task tiene un deadline; si expira, se reasigna
- `Retry`: si un worker falla, reintentar en otro (hasta N intentos)
- `RoundRobin`: dispatch simple entre workers conocidos
- `TaskId`: idempotencia — mismo task no se ejecuta dos veces

**Non-goals:** Pricing dinámico, DAG routing, auto-scaling

**Acceptance:**
- Job con 10 prompts → 10 tasks despachados correctamente
- Worker que no responde en 30s → task reasignado
- 3 reintentos fallidos → job marcado como `failed`
- Mismo task_id no se ejecuta dos veces

---

## V0-3: Multi-worker + crash recovery

**Prioridad:** Semana 2
**Label:** `v0`, `feature`

Dos workers locales (mismo modelo o distintos), round-robin dispatch, crash recovery real.

**Scope:**
- 2 workers vía Ollama (granite3.1-moe:3b + qwen3:8b)
- Round-robin dispatch entre workers
- Crash test: matar worker-0 a mitad de job → worker-1 rescata
- Heartbeat periódico workers → coordinador
- Métricas: success rate, retry rate, queue time, execution time

**Non-goals:** Más de 2 workers, red real multi-máquina, balanceo de carga

**Acceptance:**
- 50 jobs con 2 workers → ≥95% success
- Crash de worker → recovery en <30s
- Cero jobs huérfanos tras crash
- Ambos workers procesan tasks simultáneamente

---

## V0-4: Métricas E2E y benchmark reproducible

**Prioridad:** Semana 3
**Label:** `v0`, `feature`

Publicar un benchmark que compare: 1 nodo local vs 2 workers coordinados vs 2 workers con fallo inducido.

**Scope:**
- `MetricsCollector`: success_rate, retry_rate, queue_time_ms, execution_time_ms, tokens_total, cost_per_1m_tokens
- Script `scripts/bench.sh` que ejecuta el benchmark y produce un reporte
- Reporte markdown con tabla comparativa
- Publicar en `docs/benchmarks/v0-<date>.md`

**Non-goals:** Dashboard, alertas, series temporales

**Acceptance:**
- Benchmark reproducible con un solo comando
- Tabla comparativa con las 3 configuraciones
- Métricas incluyen p50/p95/p99 para latencia

---

## V0-5: Segundo runtime — validar InferencePort

**Prioridad:** Semana 4
**Label:** `v0`, `feature`

Implementar un segundo adaptador de runtime (llama.cpp o mock determinista) y pasar la misma suite de tests contractuales sin modificar el scheduler ni el job model.

**Scope:**
- `LlamaCppEngine` o `MockEngine` como segundo backend
- Misma interfaz `generate(prompt, max_tokens, seed) → (text, tokens, elapsed_ms)`
- Ejecutar exactamente el mismo `Job` con ambos backends
- Verificar que scheduler, API, y job model no requieren cambios

**Non-goals:** Optimización de rendimiento, soporte multi-GPU, más de 2 backends

**Acceptance:**
- Cambiar runtime por configuración → mismo test suite pasa
- Cero imports de vLLM u Ollama en domain/scheduler
- `InferencePort` trait definido y documentado

---

## Criterios go/no-go para V1

| Métrica | Target | Decisión si no se cumple |
|---|---|---|
| Jobs completados | ≥95% | Priorizar confiabilidad |
| Crash recovery | <30s reasignación | Corregir leases/timeouts |
| Jobs huérfanos | 0 | Rediseñar idempotencia |
| Costo por 1M tokens | ≤ API centralizada | Re evaluar caso de uso |
| Segundo runtime | Mismo test suite | Corregir abstracción |
