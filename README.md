# Smart Lightning 

## Зависимости
* **cpr** - HTTP запросы [ссылка](https://github.com/libcpr/cpr)
* **spdlog** - логирование [ссылка](https://github.com/gabime/spdlog)
* **ONNX Runtime** - для работы нейростей [ссылка](https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz). 
* **OpenCV** 

**Важно:** Для работы ONNX необходимо установить и распаковать архив. Укажите путь в 'CMakeLists.txt' в переменной 'ONNXRUNTIME_DIR'.

## Сборка проекта

1. **Клонируйте репозиторий:**
2. **Настройте CMake**
* Откройте 'CMakeLists.txt' и укажите правильный путь к ONNX Runtime.
* Создайте папку для сборки:
    '''
    mkdir build
    cd build
    cmake ..
    '''

3. **Соберите проект**
 '''make'''

4. **Запустите из корня проекта**
 '''./build/smart_lightning'''

## Конфигурация
Все настройки находятся в файле 'config.json'
