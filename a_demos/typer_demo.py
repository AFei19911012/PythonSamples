# -*- coding: utf-8 -*-
"""
 Created on 2021/6/18 22:21
 Filename   : ex_typer.py
 Author     : Taosy.W
 Zhihu      : https://www.zhihu.com/people/1105936347
 Github     : https://github.com/AFei19911012
 Description:
"""
# Source: 
# =======================================================
import typer
from typing import Optional


app = typer.Typer(help='Awesome CLI ...')


# @app.command()
# def create(username='Taosy.W'):
#     """
#     create a new user
#     """
#     typer.echo(f"Creating user: {username}")
#
#
# @app.command()
# def delete(username: str, force: bool = typer.Option(..., prompt="Are you sure you want to delete the user?", help='Force deletion without confirmation ...')):
#     """
#     delete a new user \n
#     If --force is not used, will ask for confirmation
#     """
#     if force:
#         typer.secho(f"Deleting user: {username}", fg=typer.colors.MAGENTA)
#     else:
#         typer.secho("Operation cancelled", fg=typer.colors.MAGENTA)


def hello(name='Taosy.W'):
    style_name = typer.style(name, fg=typer.colors.MAGENTA, bg=typer.colors.YELLOW, bold=True)
    typer.echo(f"Hello {style_name}")


def goodbye(name='Taosy.W', formal: bool = False):
    if formal:
        typer.echo(f"Goodbye Mr. {name}. Have a good day.")
    else:
        typer.echo(f"Bye {name}!")


@app.command()
def main(say_hello: bool = typer.Option(False, '--hello', '-h', help='Say hello ...'),
         say_goodbye: bool = typer.Option(False, '--goodbye', '-g', help='Say goodbye ...')):
    print('main process ...')

    if say_hello:
        hello()

    if say_goodbye:
        goodbye(formal=True)


if __name__ == "__main__":
    app()
