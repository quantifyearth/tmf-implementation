Methodology RFCs
----------------

A collection of methodologies in an RFC format, these can be cross-referenced
in code implementations to provide an informal argument as to the specification
of what the function does.

To build a particular RFC you can use Docker.

```
docker build . -t rfc
docker run -v $PWD/<RFC_DIR>:/drafts -it rfc

```

Open the corresponding `index.html` file in your browser and edit the `index.md` file.
Do read about [the syntax for RFC x Markdown](https://mmark.miek.nl)