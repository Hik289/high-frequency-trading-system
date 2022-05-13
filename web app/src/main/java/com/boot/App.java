package com.boot;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
@MapperScan(value="com.boot.dao")
public class App {
	public static void main(String[] args) {
		SpringApplication.run(App.class, args);
	}
}