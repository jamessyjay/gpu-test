# K8s HOWTO: GPU Acceptance Tests

Ниже — минимальная, практичная инструкция для DevOps: как прогнать тесты на всех нодах и как устроить multi-node DDP.

## Требования
- Кластер с NVIDIA GPU на нодах.
- NVIDIA Device Plugin установлен и включён.
- Доступ к образу: `ghcr.io/jamessyjay/gpu-cluster-acceptance:latest` (или свой registry).

## Быстрый старт (TL;DR)
- Прогнать на всех нодах по 1 поду: 
  ```bash
  kubectl apply -f k8s/daemonset-accept.yaml
  ```
- Multi-node DDP (несколько нод, несколько GPU на ноду):
  ```bash
  # при необходимости поправь replicas и лимит GPU в YAML
  kubectl apply -f k8s/statefulset-ddp.yaml
  ```

## Что делает контейнер
- По умолчанию запускает `python src/run_all_tests.py --quick`.
- Автодетект числа GPU на ноде и запуск обоих тестов:
  - `gpu_tests.py` (вычисление + тренировка на всех локальных GPU)
  - `ddp_tests.py` (через torchrun), если тест запускается мульти-процессно.
- Репорты пишутся в JSON, путь можно задать `--report-dir`.

## Запуск на всех нодах (DaemonSet)
- Файл: `k8s/daemonset-accept.yaml`
- Что делает:
  - Под на КАЖДОЙ ноде с GPU.
  - Запускает `run_all_tests.py --quick` с лимитом `nvidia.com/gpu: 1` на под.
  - Пишет JSON-отчёт в `/app/reports` (emptyDir, эпемеральное хранилище).
- Как посмотреть:
  ```bash
  kubectl get pods -l app=gpu-accept-daemon -o wide
  kubectl logs -l app=gpu-accept-daemon --tail=200
  # Скопировать отчёт с конкретного пода
  POD=$(kubectl get pods -l app=gpu-accept-daemon -o jsonpath='{.items[0].metadata.name}')
  kubectl cp $POD:/app/reports ./reports-daemon
  ```
- Как удалить:
  ```bash
  kubectl delete -f k8s/daemonset-accept.yaml
  ```

## Multi-node DDP (StatefulSet)
- Файл: `k8s/statefulset-ddp.yaml`
- Что делает:
  - Разворачивает N подов (реплик) и поднимает headless Service для rendezvous.
  - По умолчанию `replicas: 64`, лимит `nvidia.com/gpu: 8` на под — меняй под свой кластер.
  - Внутри запускается `torchrun --nnodes=<replicas> --nproc_per_node=<GPU-per-node> ... src/ddp_tests.py`.
- Параметры для тюнинга в YAML:
  - `spec.replicas` — число нод (NNODES).
  - `resources.limits.nvidia.com/gpu` — GPU на ноду (GPUS_PER_NODE).
- Как смотреть и собирать логи/репорты:
  ```bash
  kubectl get pods -l app=gpu-accept-ddp -o wide
  kubectl logs -l app=gpu-accept-ddp --tail=100
  # Скопировать отчёт с конкретного пода (обычно pod-0 пишет сводку)
  POD0=$(kubectl get pods -l app=gpu-accept-ddp -o jsonpath='{.items[?(@.metadata.name~=".*-0$")].metadata.name}')
  kubectl cp $POD0:/app/reports ./reports-ddp
  ```
- Как удалить:
  ```bash
  kubectl delete -f k8s/statefulset-ddp.yaml
  kubectl delete svc gpu-accept-ddp
  ```

## Изменение контейнерного образа
- Если используете приватный registry:
  - Поменяйте `image:` в YAML.
  - Добавьте `imagePullSecrets` в pod spec.

## Где смотреть результаты
- Логи подов: `kubectl logs ...` — видны `[ENV]`, `[COMPUTE]`, `[TRAIN]`, `[DDP]`, `[RESULT]`.
- JSON-репорты:
  - DaemonSet: `/app/reports/summary.json` на каждой ноде.
  - StatefulSet: `/app/reports/ddp_tests_result.json` (обычно на pod-0), плюс локальные summary от одиночных тестов при их запуске.

## Частые проблемы
- Нет GPU в поде: проверь `nvidia.com/gpu` и device plugin.
- NCCL/RDZV: убедись, что headless service доступен, порт 29500 открыт, DNS работает.
- Несовпадение версий CUDA/torch: образ должен соответствовать версии `pytorch-cuda` в conda env.

