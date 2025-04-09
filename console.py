from rich.console import Console

console = Console()


def warning(*args, **kwargs):
    console.print('[magenta][WARNING][/]', *args, **kwargs)


def error(*args, **kwargs):
    console.print('[bright_red][ERROR][/]', *args, **kwargs)


def info(*args, **kwargs):
    console.print('[cyan][INFO][/]', *args, **kwargs)
