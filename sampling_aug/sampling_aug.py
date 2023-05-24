import click


@click.group()
def main():
    """
        CLI for running experiments concerning
    """
    pass


@main.command()
def hello():
    click.echo("hello hello script")


if __name__ == '__main__':
    main()
