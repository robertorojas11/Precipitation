# PROMPT PARA CLAUDE CODE — Rollback seguro de la primera migración

## Contexto para quien ejecuta esto
Ya se corrió una primera versión del prompt de migración sobre este repositorio (el que archivaba el modelo determinista, extendía adquisición de datos, etc.). Antes de aplicar el prompt nuevo y mejorado, es necesario **revertir de forma segura** lo que ese primer prompt ya haya alcanzado a modificar en el servidor — sin asumir qué tanto llegó a ejecutar, y sin destruir nada por accidente.

Este prompt es **investigativo primero, destructivo después, y solo con confirmación explícita**. No asumas el alcance del rollback: descúbrelo.

---

## PASO 0 — Diagnóstico: ¿qué se alcanzó a modificar?

Antes de revertir nada, responde estas preguntas y repórtalas al usuario en un resumen antes de continuar:

1. `git log --oneline -30` y `git status` — ¿hay commits nuevos desde antes de la migración? ¿hay cambios sin commitear (working tree sucio)?
2. `git tag --list` — ¿existe el tag `pre-migration-v2_clean` (o similar) que el primer prompt debía crear antes de tocar nada? Si existe, esta es la ancla segura para el rollback.
3. `git branch -a` — ¿se creó alguna rama nueva relacionada con la migración?
4. `find . -maxdepth 2 -iname "legacy*"` — ¿existe ya una carpeta `legacy/` con el modelo determinista archivado?
5. `find . -path ./.git -prune -o -newer <(git log -1 --format=%cd --date=iso pre-migration-v2_clean 2>/dev/null || echo /dev/null) -type f -print 2>/dev/null | head -100` (ajustar si el tag no existe) — lista de archivos modificados/creados después del punto de partida.
6. ¿Existen carpetas nuevas de datos descargados (`raw/era5_cp_lsp`, `raw/imerg`, o similares) que ya hayan consumido tiempo/ancho de banda? Estas **no se deben borrar** aunque el código se revierta — son costosas de rehacer.
7. ¿Existen documentos nuevos en `docs/decisions/`, `docs/MIGRATION.md`, `docs/architecture_v3.md`, etc.? Estos documentan decisiones y hallazgos — evalúa si conviene conservarlos como referencia histórica incluso si el código se revierte (muévelos a `docs/archive/migration_attempt_1/` en vez de borrarlos).

**Presenta este diagnóstico completo al usuario y espera confirmación explícita antes de continuar al Paso 1.** No asumas que "revertir todo" es lo que se quiere — puede que solo haga falta revertir el código, conservando los datos ya descargados y la documentación de decisiones ya tomada.

---

## PASO 1 — Estrategia de rollback según lo que se encontró

### Caso A: existe el tag `pre-migration-v2_clean` y el working tree está limpio (todo comiteado)
Este es el caso más simple y seguro:
```bash
git checkout main   # o la rama principal correspondiente
git diff pre-migration-v2_clean HEAD --stat   # revisar el alcance del diff antes de revertir
```
Presenta el `diff --stat` al usuario. Si confirma, procede con:
```bash
git reset --hard pre-migration-v2_clean
```
**Antes de este `reset --hard`, crea una rama de respaldo con el estado actual (migrado), por si se necesita recuperar algo después:**
```bash
git branch migration-attempt-1-backup HEAD
```

### Caso B: no existe el tag, pero los commits de la migración son identificables
Si el primer prompt no llegó a crear el tag (o el usuario no siguió esa instrucción), busca el último commit *antes* de que empezara la migración (pídele al usuario el hash o la fecha aproximada si no es obvio en el log), y aplica la misma lógica del Caso A usando ese hash en vez del tag.

### Caso C: hay cambios sin commitear (working tree sucio)
No los descartes sin mostrarlos primero:
```bash
git diff
git status
```
Muestra el diff completo al usuario. Pregunta explícitamente si (a) se descartan con `git checkout -- .` / `git clean -fd` (destructivo), o (b) se guardan en un stash por si acaso: `git stash push -m "cambios de migración sin commitear, antes de rollback"`.

### Caso D: ya se movieron archivos a `legacy/` o `outputs/legacy_v2_clean_deterministic/`
Si el modelo determinista original ya fue archivado correctamente ahí, **no lo deshagas** — es exactamente el resultado correcto que buscábamos preservar, independientemente de si el resto de la migración se revierte. Verifica únicamente que el modelo archivado siga siendo ejecutable desde su nueva ubicación antes de continuar.

---

## PASO 2 — Verificación post-rollback

Después de aplicar el rollback elegido:

1. Corre la suite de tests existente: `python -m unittest discover -s tests` (o el comando equivalente que use el repo) y confirma que pasa igual que antes de la migración.
2. Corre el pipeline original end-to-end en modo de verificación rápida (si existe un modo `--fast` o similar en `validate_dataset.py` / `pipeline.py`) para confirmar que el estado revertido es funcional.
3. Compara `architecture.md`, `README.md` y `docs/` contra su versión pre-migración — confirma que reflejan correctamente el estado actual del código (no hay documentación "adelantada" describiendo algo que ya no existe).

---

## PASO 3 — Documentar el rollback (no lo dejes sin registro)

Crea (o actualiza si ya existe) `docs/MIGRATION.md` con una sección nueva:

```markdown
## Intento de migración #1 — revertido el [fecha]

Se ejecutó una primera versión del prompt de migración, que alcanzó a: [lista lo que el diagnóstico del Paso 0 encontró].

Se decidió revertir porque: [razón que dé el usuario — p.ej. "se definió una arquitectura mejor tras revisar dos papers adicionales antes de continuar"].

Estado conservado del intento 1 (no revertido):
- [lista lo que se decidió mantener: datos ya descargados, decisiones documentadas, modelo archivado en legacy/, etc.]

Rama de respaldo del intento 1: `migration-attempt-1-backup` (si aplica).

El intento 2 se documenta a partir de `docs/decisions/0002-*.md`.
```

---

## PASO 4 — Confirmación final antes de cerrar

Antes de dar por terminado el rollback, reporta al usuario:
1.Estado final de git log y git status.
2.Qué se conservó explícitamente (datos, documentación, legacy/).
3.Qué se revirtió.
4.Que el repositorio está listo para recibir el nuevo prompt de implementación (0002-implementacion-v2.md o como se llame el prompt maestro nuevo).
No apliques el prompt de nueva implementación en la misma sesión sin que el usuario confirme que el rollback quedó como se esperaba