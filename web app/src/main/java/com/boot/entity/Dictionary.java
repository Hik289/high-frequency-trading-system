package com.boot.entity;

import java.io.Serializable;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;

import lombok.Data;

/**
 *  字典
 *  标签
 */
@Entity
@Table(name = "t_dictionary")
@Data
public class Dictionary implements Serializable {
	private static final long serialVersionUID = 7989646172462159020L;
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	private Integer id;
	private String code;
	private String name;
	
	private String describe;

}
