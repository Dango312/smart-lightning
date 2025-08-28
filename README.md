# Smart Lightning 

## Зависимости
* **cpr** - HTTP запросы [ссылка](https://github.com/libcpr/cpr)
* **spdlog** - логирование [ссылка](https://github.com/gabime/spdlog)
* **ONNX Runtime** - для работы нейростей [ссылка](https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz). [cuda](https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-gpu-1.22.0.tgz)
* **OpenCV** 
* **CUDA Toolkit** [ссылка](https://developer.nvidia.com/cuda-toolkit-archive) (Тестировалось на версии 12.6)
* **cuDNN** [ссылка](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html#ubuntu-and-debian-network-installation)

**Важно:** Для работы ONNX необходимо установить и распаковать архив. Укажите путь в 'CMakeLists.txt' в переменной 'ONNXRUNTIME_DIR'.

## Сборка проекта

1. **Клонируйте репозиторий:**
2. **Настройте CMake**
* Откройте 'CMakeLists.txt' и укажите правильный путь к ONNX Runtime.
* Создайте папку для сборки:
    '''
    mkdir build
    cd build
    cmake -DONNXRUNTIME_DIR=/путь/к/onnxruntime ..      Укажите путь к версии с cpu или cuda
    '''

3. **Соберите проект**
 '''make'''

4. **Запустите из корня проекта**
 '''./build/smart_lightning'''

## Конфигурация
Все настройки находятся в файле 'config.json'


## hand_gesture_server.py (Больше не нужен!!!)
