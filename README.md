# Smart Lightning 

## Зависимости
* **cpr** - HTTP запросы https://github.com/libcpr/cpr
* **spdlog** - логирование https://github.com/gabime/spdlog?tab=readme-ov-file
* **ONNX Runtime** - для работы нейростей.
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

4. **Запустите**
 '''./smart_lightning'''

## Конфигурация
Все настройки находятся в файле 'config.json'